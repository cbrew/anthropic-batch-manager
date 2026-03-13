"""Executor: run an ExecutionPlan level-by-level."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import anthropic

from .batch_client import BatchClient
from .compiler import ExecutionPlan
from .errors import TemplateError
from .task import LLMTask, PyTask, TaskResult
from .template import render_template

logger = logging.getLogger(__name__)

# Anthropic batch API limit: 100,000 requests per batch
MAX_BATCH_SIZE = 100_000

# Below this count, use individual async API calls instead of batching
BATCH_THRESHOLD = 10


class Executor:
    """Execute a compiled plan level-by-level.

    Within each level:
    - PyTasks run sequentially first (they're fast glue code).
    - Fewer than 10 LLM tasks: individual concurrent API calls.
    - 10+ LLM tasks: submitted as Anthropic batches, chunked to stay
      within the API limit of 100k requests per batch.
    - Failed deps propagate as 'skipped' to downstream tasks.
    """

    def __init__(
        self,
        plan: ExecutionPlan,
        client: anthropic.AsyncAnthropic | None = None,
        max_batch_size: int = MAX_BATCH_SIZE,
        batch_threshold: int = BATCH_THRESHOLD,
    ) -> None:
        self._plan = plan
        self._client = client or anthropic.AsyncAnthropic()
        self._batch_client = BatchClient(client=self._client)
        self._max_batch_size = max_batch_size
        self._batch_threshold = batch_threshold

        # State
        self._resolved: dict[str, TaskResult] = {}
        self._resolved_content: dict[str, str] = {}  # task_id -> output text

    async def run(self) -> dict[str, TaskResult]:
        """Execute all levels and return results for every task."""
        tasks = self._plan.tasks

        for level in self._plan.levels:
            py_ids: list[str] = []
            llm_ids: list[str] = []

            for tid in level.task_ids:
                task = tasks[tid]

                # Skip if any dep failed
                if self._any_dep_failed(task):
                    self._resolved[tid] = TaskResult(
                        task_id=tid,
                        status="skipped",
                        error="Dependency failed",
                    )
                    continue

                if isinstance(task, PyTask):
                    py_ids.append(tid)
                else:
                    llm_ids.append(tid)

            # Run PyTasks sequentially
            for tid in py_ids:
                await self._run_pytask(tid, tasks[tid])

            # Dispatch LLM tasks
            if llm_ids:
                await self._run_llm_level(llm_ids, tasks)

        return dict(self._resolved)

    def _any_dep_failed(self, task: LLMTask | PyTask) -> bool:
        """Check if any dependency errored or was skipped."""
        return any(
            dep_id in self._resolved
            and self._resolved[dep_id].status in ("errored", "skipped")
            for dep_id in task.deps
        )

    async def _run_pytask(self, task_id: str, task: PyTask) -> None:
        """Run a single PyTask."""
        logger.info("Running PyTask: %s", task_id)
        try:
            dep_results = {
                dep_id: self._resolved_content.get(dep_id, "")
                for dep_id in task.deps
            }
            result = task.fn(dep_results)
            content = str(result)
            self._resolved[task_id] = TaskResult(
                task_id=task_id, status="succeeded", content=content
            )
            self._resolved_content[task_id] = content
        except Exception as e:
            logger.error("PyTask %s failed: %s", task_id, e)
            self._resolved[task_id] = TaskResult(
                task_id=task_id, status="errored", error=str(e)
            )

    def _build_params(self, task: LLMTask) -> dict[str, Any]:
        """Build API parameters dict for an LLM task."""
        messages = [{"role": "user", "content": self._resolve_prompt(task)}]
        params: dict[str, Any] = {
            "model": task.model,
            "max_tokens": task.max_tokens,
            "messages": messages,
        }
        if task.system:
            params["system"] = task.system
        if task.temperature is not None:
            params["temperature"] = task.temperature
        return params

    async def _run_llm_level(
        self, task_ids: list[str], tasks: dict[str, Any]
    ) -> None:
        """Dispatch LLM tasks — individual calls if few, batch API if many."""
        # Resolve prompts and build params, filtering out template errors
        prepared: list[tuple[str, dict[str, Any]]] = []
        for tid in task_ids:
            task: LLMTask = tasks[tid]
            try:
                params = self._build_params(task)
            except TemplateError as e:
                logger.error("Template error for task %s: %s", tid, e)
                self._resolved[tid] = TaskResult(
                    task_id=tid, status="errored", error=str(e)
                )
                continue
            prepared.append((tid, params))

        if not prepared:
            return

        if len(prepared) < self._batch_threshold:
            await self._run_llm_individual(prepared)
        else:
            await self._run_llm_batched(prepared)

    async def _run_llm_individual(
        self, prepared: list[tuple[str, dict[str, Any]]]
    ) -> None:
        """Run LLM tasks as concurrent individual API calls."""
        logger.info("Running %d LLM tasks as individual calls", len(prepared))

        async def call_one(tid: str, params: dict[str, Any]) -> None:
            try:
                message = await self._client.messages.create(**params)
                content = ""
                if message.content:
                    content = message.content[0].text
                usage_dict = None
                if message.usage:
                    usage_dict = {
                        "input_tokens": message.usage.input_tokens,
                        "output_tokens": message.usage.output_tokens,
                    }
                self._resolved[tid] = TaskResult(
                    task_id=tid,
                    status="succeeded",
                    content=content,
                    usage=usage_dict,
                )
                self._resolved_content[tid] = content
            except Exception as e:
                logger.error("LLM call for task %s failed: %s", tid, e)
                self._resolved[tid] = TaskResult(
                    task_id=tid, status="errored", error=str(e)
                )

        await asyncio.gather(*(call_one(tid, params) for tid, params in prepared))

    async def _run_llm_batched(
        self, prepared: list[tuple[str, dict[str, Any]]]
    ) -> None:
        """Submit LLM tasks via Batch API, chunked for API limits."""
        batch_items = [
            {"custom_id": tid, "params": params} for tid, params in prepared
        ]

        for chunk_start in range(0, len(batch_items), self._max_batch_size):
            chunk = batch_items[chunk_start : chunk_start + self._max_batch_size]

            logger.info(
                "Submitting batch with %d LLM tasks (chunk %d/%d)",
                len(chunk),
                chunk_start // self._max_batch_size + 1,
                (len(batch_items) + self._max_batch_size - 1) // self._max_batch_size,
            )

            batch_id = await self._batch_client.submit_batch(chunk)
            await self._batch_client.poll_until_done(batch_id)
            results = await self._batch_client.get_results(batch_id)

            for tid, result in results.items():
                self._resolved[tid] = result
                if result.content:
                    self._resolved_content[tid] = result.content

    def _resolve_prompt(self, task: LLMTask) -> str:
        """Get the final prompt string for an LLM task."""
        if task.prompt is not None:
            return task.prompt
        assert task.prompt_template is not None
        return render_template(task.prompt_template, self._resolved_content)
