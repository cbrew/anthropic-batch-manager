"""Tests for the executor with mock batch and individual API clients."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from batch_compiler.compiler import compile_graph
from batch_compiler.executor import Executor
from batch_compiler.graph import TaskGraph
from batch_compiler.task import LLMTask, PyTask, TaskResult


def _make_mock_batch_client(responses: dict[str, str]):
    """Create a mock BatchClient that returns canned responses."""
    mock = MagicMock()
    mock.submit_batch = AsyncMock(return_value="batch-123")
    mock.poll_until_done = AsyncMock()

    async def mock_get_results(batch_id: str) -> dict[str, TaskResult]:
        return {
            tid: TaskResult(task_id=tid, status="succeeded", content=text)
            for tid, text in responses.items()
        }

    mock.get_results = AsyncMock(side_effect=mock_get_results)
    return mock


def _make_mock_anthropic_client(responses: dict[str, str]):
    """Create a mock AsyncAnthropic whose messages.create returns canned responses.

    Uses the model param to look up the custom_id isn't available, so we
    key off the prompt text (first user message content).
    """
    mock_client = MagicMock()

    async def mock_create(**kwargs):
        # Find which task this is by matching prompt content
        prompt = kwargs["messages"][0]["content"]
        # Look up by prompt text in responses
        for tid, text in responses.items():
            if tid in prompt or prompt == tid:
                msg = MagicMock()
                msg.content = [MagicMock(text=text)]
                msg.usage = MagicMock(input_tokens=10, output_tokens=20)
                return msg
        # Fallback: return first response
        tid, text = next(iter(responses.items()))
        msg = MagicMock()
        msg.content = [MagicMock(text=text)]
        msg.usage = MagicMock(input_tokens=10, output_tokens=20)
        return msg

    mock_client.messages = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=mock_create)
    return mock_client


class TestExecutorIndividualCalls:
    """Tests for the individual async call path (< batch_threshold tasks)."""

    @pytest.mark.asyncio
    async def test_single_task_uses_individual_call(self):
        g = TaskGraph()
        g.add(LLMTask(id="t1", prompt="t1"))
        plan = compile_graph(g)

        mock_client = _make_mock_anthropic_client({"t1": "World"})
        executor = Executor(plan=plan, client=mock_client)

        results = await executor.run()
        assert results["t1"].status == "succeeded"
        assert results["t1"].content == "World"
        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_few_tasks_use_individual_calls(self):
        """< 10 tasks should use individual calls, not batch API."""
        g = TaskGraph()
        for i in range(5):
            g.add(LLMTask(id=f"t{i}", prompt=f"t{i}"))
        plan = compile_graph(g)

        responses = {f"t{i}": f"Result {i}" for i in range(5)}
        mock_client = _make_mock_anthropic_client(responses)
        executor = Executor(plan=plan, client=mock_client)

        results = await executor.run()
        assert all(results[f"t{i}"].status == "succeeded" for i in range(5))
        assert mock_client.messages.create.call_count == 5

    @pytest.mark.asyncio
    async def test_individual_call_error_handling(self):
        g = TaskGraph()
        g.add(LLMTask(id="t1", prompt="t1"))
        g.add(LLMTask(id="t2", prompt_template="Use {t1}", deps=["t1"]))
        plan = compile_graph(g)

        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(
            side_effect=Exception("API down")
        )

        executor = Executor(plan=plan, client=mock_client)
        results = await executor.run()
        assert results["t1"].status == "errored"
        assert results["t1"].error.type == "Exception"
        assert "API down" in results["t1"].error.message
        assert results["t2"].status == "skipped"

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
    async def test_individual_llm_then_pytask(self):
        """Individual call result feeds into downstream PyTask."""
        g = TaskGraph()
        g.add(LLMTask(id="llm1", prompt="llm1"))
        g.add(PyTask(
            id="py1",
            fn=lambda deps: f"processed: {deps['llm1']}",
            deps=["llm1"],
        ))
        plan = compile_graph(g)

        mock_client = _make_mock_anthropic_client({"llm1": "Answer"})
        executor = Executor(plan=plan, client=mock_client)

        results = await executor.run()
        assert results["llm1"].content == "Answer"
        assert results["py1"].content == "processed: Answer"

    @pytest.mark.asyncio
    async def test_two_level_individual_calls(self):
        """Two levels, each below threshold — both use individual calls."""
        g = TaskGraph()
        g.add(LLMTask(id="a", prompt="a"))
        g.add(LLMTask(id="b", prompt="b", deps=["a"]))
        plan = compile_graph(g)

        responses = {"a": "R1", "b": "R2"}
        mock_client = _make_mock_anthropic_client(responses)
        executor = Executor(plan=plan, client=mock_client)

        results = await executor.run()
        assert results["a"].content == "R1"
        assert results["b"].content == "R2"
        assert mock_client.messages.create.call_count == 2


class TestExecutorBatchPath:
    """Tests for the batch API path (>= batch_threshold tasks)."""

    @pytest.mark.asyncio
    async def test_batch_path_used_above_threshold(self):
        """Tasks >= threshold use batch API."""
        g = TaskGraph()
        for i in range(5):
            g.add(LLMTask(id=f"t{i}", prompt=f"Task {i}"))
        plan = compile_graph(g)

        responses = {f"t{i}": f"Result {i}" for i in range(5)}
        mock_batch = _make_mock_batch_client(responses)

        # Set threshold=3 so 5 tasks use batch path
        executor = Executor(plan=plan, batch_threshold=3)
        executor._batch_client = mock_batch

        results = await executor.run()
        assert all(results[f"t{i}"].status == "succeeded" for i in range(5))
        mock_batch.submit_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_chunking_large_batch(self):
        """Batches exceeding max_batch_size are chunked."""
        g = TaskGraph()
        for i in range(5):
            g.add(LLMTask(id=f"t{i}", prompt=f"Task {i}"))
        plan = compile_graph(g)

        responses = {f"t{i}": f"Result {i}" for i in range(5)}
        mock_batch = _make_mock_batch_client(responses)

        # threshold=2 → batch path; max_batch_size=2 → 3 chunks
        executor = Executor(plan=plan, batch_threshold=2, max_batch_size=2)
        executor._batch_client = mock_batch

        results = await executor.run()
        assert all(results[f"t{i}"].status == "succeeded" for i in range(5))
        assert mock_batch.submit_batch.call_count == 3

    @pytest.mark.asyncio
    async def test_two_level_batch_pipeline(self):
        """Two levels, each at/above threshold — both use batch API."""
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

        mock_batch = MagicMock()
        mock_batch.submit_batch = AsyncMock(return_value="batch-x")
        mock_batch.poll_until_done = AsyncMock()
        mock_batch.get_results = AsyncMock(side_effect=mock_get_results)

        # threshold=1 so even single tasks use batch path
        executor = Executor(plan=plan, batch_threshold=1)
        executor._batch_client = mock_batch

        results = await executor.run()
        assert results["a"].content == "R1"
        assert results["b"].content == "R2"
        assert mock_batch.submit_batch.call_count == 2

    @pytest.mark.asyncio
    async def test_failed_dep_skips_downstream(self):
        g = TaskGraph()
        g.add(LLMTask(id="a", prompt="Will fail"))
        g.add(LLMTask(id="b", prompt_template="Use {a}", deps=["a"]))
        plan = compile_graph(g)

        async def mock_get_results(batch_id):
            from batch_compiler.task import TaskError
            return {
                "a": TaskResult(task_id="a", status="errored", error=TaskError(type="api_error", message="API error"))
            }

        mock_batch = MagicMock()
        mock_batch.submit_batch = AsyncMock(return_value="batch-x")
        mock_batch.poll_until_done = AsyncMock()
        mock_batch.get_results = AsyncMock(side_effect=mock_get_results)

        executor = Executor(plan=plan, batch_threshold=1)
        executor._batch_client = mock_batch

        results = await executor.run()
        assert results["a"].status == "errored"
        assert results["b"].status == "skipped"
