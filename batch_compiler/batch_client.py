"""Async wrapper around the Anthropic Batch API."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import anthropic

from .errors import BatchError
from .task import TaskError, TaskResult

logger = logging.getLogger(__name__)


class BatchClient:
    """Thin async client for creating, polling, and retrieving Anthropic batches."""

    def __init__(
        self,
        client: anthropic.AsyncAnthropic | None = None,
        poll_initial_delay: float = 5.0,
        poll_max_delay: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self._client = client or anthropic.AsyncAnthropic()
        self._poll_initial_delay = poll_initial_delay
        self._poll_max_delay = poll_max_delay
        self._max_retries = max_retries

    async def submit_batch(
        self, items: list[dict[str, Any]]
    ) -> str:
        """Create a batch and return its id.

        Args:
            items: List of {custom_id, params} dicts for the batch.

        Returns:
            The batch id string.

        Raises:
            BatchError: If batch creation fails after retries, or immediately
                for non-retryable errors (auth, bad request).
        """
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                batch = await self._client.messages.batches.create(requests=items)
                logger.info("Batch created: %s (%d items)", batch.id, len(items))
                return batch.id
            except anthropic.AuthenticationError as e:
                raise BatchError(
                    f"Authentication failed: {e}. Check your API key."
                ) from e
            except anthropic.PermissionDeniedError as e:
                raise BatchError(
                    f"Permission denied: {e}. Check API key permissions."
                ) from e
            except anthropic.BadRequestError as e:
                raise BatchError(
                    f"Bad request (will not retry): {e}"
                ) from e
            except Exception as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    delay = 2 ** (attempt + 1)
                    logger.warning(
                        "Batch creation attempt %d failed (%s): %s. Retrying in %ds...",
                        attempt + 1,
                        type(e).__name__,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
        raise BatchError(
            f"Batch creation failed after {self._max_retries} attempts: {last_error}"
        ) from last_error

    async def poll_until_done(self, batch_id: str) -> None:
        """Poll a batch until its processing_status is 'ended'.

        Uses exponential backoff from poll_initial_delay to poll_max_delay.

        Raises:
            BatchError: If polling encounters an unexpected error.
        """
        delay = self._poll_initial_delay
        while True:
            try:
                batch = await self._client.messages.batches.retrieve(batch_id)
            except Exception as e:
                raise BatchError(
                    f"Error polling batch {batch_id}: {e}"
                ) from e

            logger.info(
                "Batch %s: status=%s, counts=%s",
                batch_id,
                batch.processing_status,
                batch.request_counts,
            )

            if batch.processing_status == "ended":
                return

            await asyncio.sleep(delay)
            delay = min(delay * 1.5, self._poll_max_delay)

    def _extract_text(self, message) -> str:
        """Extract text content from a Message, handling non-text block types."""
        if not message.content:
            return ""
        texts = []
        for block in message.content:
            if hasattr(block, "text"):
                texts.append(block.text)
            else:
                logger.warning(
                    "Skipping non-text content block of type %s",
                    getattr(block, "type", type(block).__name__),
                )
        return "\n".join(texts)

    async def get_results(self, batch_id: str) -> dict[str, TaskResult]:
        """Retrieve results for a completed batch.

        Returns:
            Dict mapping custom_id -> TaskResult.
        """
        results: dict[str, TaskResult] = {}
        counts = {"succeeded": 0, "errored": 0, "expired": 0, "canceled": 0}

        try:
            async for item in await self._client.messages.batches.results(batch_id):
                custom_id = item.custom_id
                result_obj = item.result

                if result_obj.type == "succeeded":
                    counts["succeeded"] += 1
                    message = result_obj.message
                    content = self._extract_text(message)
                    stop_reason = message.stop_reason

                    if stop_reason == "refusal":
                        logger.warning(
                            "Task %s: model refused to respond", custom_id,
                        )
                    elif stop_reason == "max_tokens":
                        logger.warning(
                            "Task %s: output truncated (max_tokens)", custom_id,
                        )

                    usage_dict = None
                    if message.usage:
                        usage_dict = {
                            "input_tokens": message.usage.input_tokens,
                            "output_tokens": message.usage.output_tokens,
                        }

                    results[custom_id] = TaskResult(
                        task_id=custom_id,
                        status="succeeded",
                        content=content,
                        stop_reason=stop_reason,
                        usage=usage_dict,
                    )
                elif result_obj.type == "errored":
                    counts["errored"] += 1
                    error_response = result_obj.error
                    if error_response and hasattr(error_response, "error"):
                        error = TaskError(
                            type=error_response.error.type,
                            message=error_response.error.message,
                        )
                    else:
                        error = TaskError(
                            type="unknown",
                            message="No error details returned by API",
                        )
                    logger.error(
                        "Task %s errored: [%s] %s",
                        custom_id, error.type, error.message,
                    )
                    results[custom_id] = TaskResult(
                        task_id=custom_id,
                        status="errored",
                        error=error,
                    )
                else:
                    # expired or canceled
                    counts[result_obj.type] = counts.get(result_obj.type, 0) + 1
                    error = TaskError(
                        type=result_obj.type,
                        message=f"Request {result_obj.type}",
                    )
                    logger.error(
                        "Task %s: %s", custom_id, result_obj.type,
                    )
                    results[custom_id] = TaskResult(
                        task_id=custom_id,
                        status="errored",
                        error=error,
                    )
        except Exception as e:
            raise BatchError(
                f"Error retrieving results for batch {batch_id}: {e}"
            ) from e

        logger.info(
            "Batch %s results: %d succeeded, %d errored, %d expired, %d canceled",
            batch_id,
            counts["succeeded"],
            counts["errored"],
            counts["expired"],
            counts["canceled"],
        )
        if counts["errored"] or counts["expired"] or counts["canceled"]:
            logger.warning(
                "Batch %s had %d non-success results",
                batch_id,
                counts["errored"] + counts["expired"] + counts["canceled"],
            )

        return results
