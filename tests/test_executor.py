"""Tests for the executor with a mock batch client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from batch_compiler.compiler import compile_graph
from batch_compiler.executor import Executor
from batch_compiler.graph import TaskGraph
from batch_compiler.task import LLMTask, PyTask, TaskResult


def _make_mock_batch_client(responses: dict[str, str]):
    """Create a mock BatchClient that returns canned responses.

    Args:
        responses: dict of task_id -> response content text.
    """
    mock = MagicMock()
    mock.submit_batch = AsyncMock(return_value="batch-123")
    mock.poll_until_done = AsyncMock()

    async def mock_get_results(batch_id: str) -> dict[str, TaskResult]:
        # Return results for whatever was submitted
        return {
            tid: TaskResult(task_id=tid, status="succeeded", content=text)
            for tid, text in responses.items()
        }

    mock.get_results = AsyncMock(side_effect=mock_get_results)
    return mock


class TestExecutor:
    @pytest.mark.asyncio
    async def test_single_llm_task(self):
        g = TaskGraph()
        g.add(LLMTask(id="t1", prompt="Hello"))
        plan = compile_graph(g)

        mock_client = _make_mock_batch_client({"t1": "World"})
        executor = Executor(plan=plan)
        executor._batch_client = mock_client

        results = await executor.run()
        assert results["t1"].status == "succeeded"
        assert results["t1"].content == "World"
        mock_client.submit_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_pytask_only(self):
        g = TaskGraph()
        g.add(PyTask(id="py1", fn=lambda deps: "computed"))
        plan = compile_graph(g)

        executor = Executor(plan=plan)
        results = await executor.run()
        assert results["py1"].status == "succeeded"
        assert results["py1"].content == "computed"

    @pytest.mark.asyncio
    async def test_llm_then_pytask(self):
        g = TaskGraph()
        g.add(LLMTask(id="llm1", prompt="Question"))
        g.add(PyTask(
            id="py1",
            fn=lambda deps: f"processed: {deps['llm1']}",
            deps=["llm1"],
        ))
        plan = compile_graph(g)

        mock_client = _make_mock_batch_client({"llm1": "Answer"})
        executor = Executor(plan=plan)
        executor._batch_client = mock_client

        results = await executor.run()
        assert results["llm1"].content == "Answer"
        assert results["py1"].content == "processed: Answer"

    @pytest.mark.asyncio
    async def test_fan_out_batching(self):
        """All independent tasks at level 0 go in one batch."""
        g = TaskGraph()
        for i in range(5):
            g.add(LLMTask(id=f"t{i}", prompt=f"Task {i}"))
        plan = compile_graph(g)

        responses = {f"t{i}": f"Result {i}" for i in range(5)}
        mock_client = _make_mock_batch_client(responses)

        executor = Executor(plan=plan)
        executor._batch_client = mock_client

        results = await executor.run()
        assert all(results[f"t{i}"].status == "succeeded" for i in range(5))
        mock_client.submit_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_chunking_large_batch(self):
        """Batches exceeding max_batch_size are chunked."""
        g = TaskGraph()
        for i in range(5):
            g.add(LLMTask(id=f"t{i}", prompt=f"Task {i}"))
        plan = compile_graph(g)

        responses = {f"t{i}": f"Result {i}" for i in range(5)}
        mock_client = _make_mock_batch_client(responses)

        # Set max_batch_size=2 so 5 tasks require 3 chunks
        executor = Executor(plan=plan, max_batch_size=2)
        executor._batch_client = mock_client

        results = await executor.run()
        assert all(results[f"t{i}"].status == "succeeded" for i in range(5))
        assert mock_client.submit_batch.call_count == 3

    @pytest.mark.asyncio
    async def test_two_level_pipeline(self):
        g = TaskGraph()
        g.add(LLMTask(id="a", prompt="Step 1"))
        g.add(LLMTask(id="b", prompt_template="Step 2: {a}", deps=["a"]))
        plan = compile_graph(g)

        call_count = 0

        async def mock_get_results(batch_id):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"a": TaskResult(task_id="a", status="succeeded", content="R1")}
            else:
                return {"b": TaskResult(task_id="b", status="succeeded", content="R2")}

        mock_client = MagicMock()
        mock_client.submit_batch = AsyncMock(return_value="batch-x")
        mock_client.poll_until_done = AsyncMock()
        mock_client.get_results = AsyncMock(side_effect=mock_get_results)

        executor = Executor(plan=plan)
        executor._batch_client = mock_client

        results = await executor.run()
        assert results["a"].content == "R1"
        assert results["b"].content == "R2"
        assert mock_client.submit_batch.call_count == 2

    @pytest.mark.asyncio
    async def test_failed_dep_skips_downstream(self):
        g = TaskGraph()
        g.add(LLMTask(id="a", prompt="Will fail"))
        g.add(LLMTask(id="b", prompt_template="Use {a}", deps=["a"]))
        plan = compile_graph(g)

        async def mock_get_results(batch_id):
            return {
                "a": TaskResult(task_id="a", status="errored", error="API error")
            }

        mock_client = MagicMock()
        mock_client.submit_batch = AsyncMock(return_value="batch-x")
        mock_client.poll_until_done = AsyncMock()
        mock_client.get_results = AsyncMock(side_effect=mock_get_results)

        executor = Executor(plan=plan)
        executor._batch_client = mock_client

        results = await executor.run()
        assert results["a"].status == "errored"
        assert results["b"].status == "skipped"
