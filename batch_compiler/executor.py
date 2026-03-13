"""Executor: run an ExecutionPlan level-by-level."""

from __future__ import annotations

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


class Executor:
    """Execute a compiled plan level-by-level.

    Within each level:
    - PyTasks run sequentially first (they're fast glue code).
    - LLM tasks are submitted as Anthropic batches, chunked to stay within
      the API limit of 100k requests per batch.
    - Failed deps propagate as 'skipped' to downstream tasks.
    """

    def __init__(
        self,
        plan: ExecutionPlan,
        client: anthropic.AsyncAnthropic | None = None,
        max_batch_size: int = MAX_BATCH_SIZE,
    ) -> None:
        self._plan = plan
        self._batch_client = BatchClient(client=client)
        self._max_batch_size = max_batch_size

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

            # Submit LLM tasks as batches (chunked for API limits)
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

    async def _run_llm_level(
        self, task_ids: list[str], tasks: dict[str, Any]
    ) -> None:
        """Build batch items for all LLM tasks in a level, chunk, and submit."""
        batch_items: list[tuple[str, dict[str, Any]]] = []

        for tid in task_ids:
            task: LLMTask = tasks[tid]
            try:
                prompt = self._resolve_prompt(task)
            except TemplateError as e:
                logger.error("Template error for task %s: %s", tid, e)
                self._resolved[tid] = TaskResult(
                    task_id=tid, status="errored", error=str(e)
                )
                continue

            messages = [{"role": "user", "content": prompt}]
            params: dict[str, Any] = {
                "model": task.model,
                "max_tokens": task.max_tokens,
                "messages": messages,
            }
            if task.system:
                params["system"] = task.system
            if task.temperature is not None:
                params["temperature"] = task.temperature

            batch_items.append((tid, {"custom_id": tid, "params": params}))

        if not batch_items:
            return

        # Chunk into batches of max_batch_size
        for chunk_start in range(0, len(batch_items), self._max_batch_size):
            chunk = batch_items[chunk_start : chunk_start + self._max_batch_size]
            items = [item for _, item in chunk]

            logger.info(
                "Submitting batch with %d LLM tasks (chunk %d/%d)",
                len(items),
                chunk_start // self._max_batch_size + 1,
                (len(batch_items) + self._max_batch_size - 1) // self._max_batch_size,
            )

            batch_id = await self._batch_client.submit_batch(items)
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
