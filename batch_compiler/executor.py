"""Executor: run an ExecutionPlan using k-threshold batching."""

from __future__ import annotations

import logging
from typing import Any

import anthropic

from .batch_client import BatchClient
from .compiler import ExecutionPlan
from .errors import TemplateError
from .graph import TaskGraph
from .task import LLMTask, PyTask, TaskResult
from .template import render_template

logger = logging.getLogger(__name__)


class Executor:
    """Execute a compiled plan, batching LLM calls and running PyTasks sequentially."""

    def __init__(
        self,
        plan: ExecutionPlan,
        graph: TaskGraph,
        batch_threshold: int = 10,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self._plan = plan
        self._graph = graph
        self._batch_threshold = batch_threshold
        self._batch_client = BatchClient(client=client)

        # State
        self._resolved: dict[str, TaskResult] = {}
        self._resolved_content: dict[str, str] = {}  # task_id -> output text

    async def run(self) -> dict[str, TaskResult]:
        """Execute the full plan and return results for all tasks."""
        tasks = self._plan.tasks

        # Build dependency tracking
        remaining: set[str] = set(tasks.keys())
        dependents: dict[str, list[str]] = {tid: [] for tid in tasks}
        for tid, task in tasks.items():
            for dep_id in task.deps:
                dependents[dep_id].append(tid)

        # Find initially ready tasks (no deps)
        ready_llm: list[str] = []
        ready_py: list[str] = []

        for tid in remaining:
            task = tasks[tid]
            if not task.deps:
                if isinstance(task, LLMTask):
                    ready_llm.append(tid)
                else:
                    ready_py.append(tid)

        while remaining:
            progress = False

            # 1. Run ready PyTasks sequentially
            while ready_py:
                py_id = ready_py.pop(0)
                if py_id not in remaining:
                    continue
                await self._run_pytask(py_id, tasks[py_id])
                remaining.discard(py_id)
                progress = True

                # Check if this unlocked new tasks
                for dep_tid in dependents[py_id]:
                    self._enqueue_or_skip(
                        dep_tid, tasks, remaining, ready_llm, ready_py
                    )

            # 2. Batch LLM tasks when we have enough (or it's all that's left)
            if ready_llm:
                # Flush if >= k, or if no more PyTasks can run and these are
                # the only way to make progress
                if len(ready_llm) >= self._batch_threshold or not ready_py:
                    batch_ids = list(ready_llm)
                    ready_llm.clear()

                    await self._run_llm_batch(batch_ids, tasks)

                    for tid in batch_ids:
                        remaining.discard(tid)
                    progress = True

                    # Check for newly ready tasks
                    for tid in batch_ids:
                        for dep_tid in dependents[tid]:
                            self._enqueue_or_skip(
                                dep_tid, tasks, remaining, ready_llm, ready_py
                            )

            if not progress:
                # No progress possible — remaining tasks have unresolvable deps
                for tid in remaining:
                    self._resolved[tid] = TaskResult(
                        task_id=tid,
                        status="skipped",
                        error="Dependencies could not be resolved",
                    )
                break

        return dict(self._resolved)

    def _all_deps_resolved(self, task: LLMTask | PyTask) -> bool:
        return all(dep_id in self._resolved for dep_id in task.deps)

    def _any_dep_failed(self, task: LLMTask | PyTask) -> bool:
        """Check if any dependency errored or was skipped."""
        return any(
            dep_id in self._resolved
            and self._resolved[dep_id].status in ("errored", "skipped")
            for dep_id in task.deps
        )

    def _enqueue_or_skip(
        self,
        tid: str,
        tasks: dict[str, Any],
        remaining: set[str],
        ready_llm: list[str],
        ready_py: list[str],
    ) -> None:
        """Add a ready task to the appropriate queue, or skip if deps failed."""
        if tid not in remaining:
            return
        task = tasks[tid]
        if not self._all_deps_resolved(task):
            return
        if self._any_dep_failed(task):
            self._resolved[tid] = TaskResult(
                task_id=tid,
                status="skipped",
                error="Dependency failed",
            )
            remaining.discard(tid)
            return
        if isinstance(task, LLMTask):
            ready_llm.append(tid)
        else:
            ready_py.append(tid)

    async def _run_pytask(
        self, task_id: str, task: PyTask  # type: ignore[type-arg]
    ) -> None:
        """Run a single PyTask sequentially."""
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

    async def _run_llm_batch(
        self, task_ids: list[str], tasks: dict[str, Any]
    ) -> None:
        """Build and submit a batch of LLM tasks, poll, and collect results."""
        batch_items = []
        skipped: list[str] = []

        for tid in task_ids:
            task: LLMTask = tasks[tid]
            try:
                prompt = self._resolve_prompt(task)
            except TemplateError as e:
                logger.error("Template error for task %s: %s", tid, e)
                self._resolved[tid] = TaskResult(
                    task_id=tid, status="errored", error=str(e)
                )
                skipped.append(tid)
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

            batch_items.append({"custom_id": tid, "params": params})

        if not batch_items:
            return

        logger.info("Submitting batch with %d LLM tasks", len(batch_items))

        batch_id = await self._batch_client.submit_batch(batch_items)
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
