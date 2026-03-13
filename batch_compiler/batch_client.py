"""Async wrapper around the Anthropic Batch API."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import anthropic

from .errors import BatchError
from .task import TaskResult

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
            BatchError: If batch creation fails after retries.
        """
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                batch = await self._client.messages.batches.create(requests=items)
                logger.info("Batch created: %s (%d items)", batch.id, len(items))
                return batch.id
            except Exception as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    delay = 2 ** (attempt + 1)
                    logger.warning(
                        "Batch creation attempt %d failed: %s. Retrying in %ds...",
                        attempt + 1,
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

    async def get_results(self, batch_id: str) -> dict[str, TaskResult]:
        """Retrieve results for a completed batch.

        Returns:
            Dict mapping custom_id -> TaskResult.
        """
        results: dict[str, TaskResult] = {}

        try:
            async for item in await self._client.messages.batches.results(batch_id):
                custom_id = item.custom_id
                result_obj = item.result

                if result_obj.type == "succeeded":
                    message = result_obj.message
                    content = ""
                    if message.content:
                        content = message.content[0].text

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
                        usage=usage_dict,
                    )
                elif result_obj.type == "errored":
                    error_msg = str(result_obj.error) if result_obj.error else "Unknown error"
                    results[custom_id] = TaskResult(
                        task_id=custom_id,
                        status="errored",
                        error=error_msg,
                    )
                else:
                    # expired or canceled
                    results[custom_id] = TaskResult(
                        task_id=custom_id,
                        status="errored",
                        error=f"Request {result_obj.type}",
                    )
        except Exception as e:
            raise BatchError(
                f"Error retrieving results for batch {batch_id}: {e}"
            ) from e

        return results
