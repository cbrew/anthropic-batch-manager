"""Tests for TaskGraph: construction, validation, and topological sort."""

import pytest

from batch_compiler.errors import CycleError, DependencyError, DuplicateTaskError
from batch_compiler.graph import TaskGraph
from batch_compiler.task import LLMTask, PyTask


class TestTaskGraphConstruction:
    def test_empty_graph(self):
        g = TaskGraph()
        assert len(g) == 0
        assert g.task_ids == set()

    def test_add_single_task(self):
        g = TaskGraph()
        g.add(LLMTask(id="t1", prompt="Hi"))
        assert len(g) == 1
        assert "t1" in g.task_ids

    def test_add_returns_self_for_chaining(self):
        g = TaskGraph()
        result = g.add(LLMTask(id="t1", prompt="Hi"))
        assert result is g

    def test_duplicate_id_raises(self):
        g = TaskGraph()
        g.add(LLMTask(id="t1", prompt="Hi"))
        with pytest.raises(DuplicateTaskError, match="t1"):
            g.add(LLMTask(id="t1", prompt="Bye"))

    def test_mixed_task_types(self):
        g = TaskGraph()
        g.add(LLMTask(id="llm1", prompt="Hi"))
        g.add(PyTask(id="py1", fn=lambda x: x, deps=["llm1"]))
        assert len(g) == 2


class TestTaskGraphValidation:
    def test_valid_linear_chain(self):
        g = TaskGraph()
        g.add(LLMTask(id="a", prompt="1"))
        g.add(LLMTask(id="b", prompt_template="{a}", deps=["a"]))
        g.add(LLMTask(id="c", prompt_template="{b}", deps=["b"]))
        g.validate()  # should not raise

    def test_missing_dependency(self):
        g = TaskGraph()
        g.add(LLMTask(id="b", prompt="Hi", deps=["a"]))
        with pytest.raises(DependencyError, match="'a'"):
            g.validate()

    def test_simple_cycle(self):
        g = TaskGraph()
        g.add(LLMTask(id="a", prompt="1", deps=["b"]))
        g.add(LLMTask(id="b", prompt="2", deps=["a"]))
        with pytest.raises(CycleError, match="Cycle"):
            g.validate()

    def test_self_cycle(self):
        g = TaskGraph()
        g.add(LLMTask(id="a", prompt="1", deps=["a"]))
        with pytest.raises(CycleError, match="Cycle"):
            g.validate()

    def test_three_node_cycle(self):
        g = TaskGraph()
        g.add(LLMTask(id="a", prompt="1", deps=["c"]))
        g.add(LLMTask(id="b", prompt="2", deps=["a"]))
        g.add(LLMTask(id="c", prompt="3", deps=["b"]))
        with pytest.raises(CycleError):
            g.validate()

    def test_diamond_dag_is_valid(self):
        g = TaskGraph()
        g.add(LLMTask(id="root", prompt="start"))
        g.add(LLMTask(id="left", prompt="l", deps=["root"]))
        g.add(LLMTask(id="right", prompt="r", deps=["root"]))
        g.add(LLMTask(id="join", prompt="j", deps=["left", "right"]))
        g.validate()  # diamond is fine


class TestTopologicalSort:
    def test_linear_chain(self):
        g = TaskGraph()
        g.add(LLMTask(id="a", prompt="1"))
        g.add(LLMTask(id="b", prompt="2", deps=["a"]))
        g.add(LLMTask(id="c", prompt="3", deps=["b"]))
        order = g.topological_sort()
        assert order.index("a") < order.index("b") < order.index("c")

    def test_independent_tasks(self):
        g = TaskGraph()
        g.add(LLMTask(id="a", prompt="1"))
        g.add(LLMTask(id="b", prompt="2"))
        g.add(LLMTask(id="c", prompt="3"))
        order = g.topological_sort()
        assert set(order) == {"a", "b", "c"}

    def test_fan_out_fan_in(self):
        g = TaskGraph()
        g.add(LLMTask(id="root", prompt="start"))
        for i in range(5):
            g.add(LLMTask(id=f"mid-{i}", prompt=f"m{i}", deps=["root"]))
        g.add(PyTask(
            id="join",
            fn=lambda x: x,
            deps=[f"mid-{i}" for i in range(5)],
        ))
        order = g.topological_sort()
        assert order[0] == "root"
        assert order[-1] == "join"
