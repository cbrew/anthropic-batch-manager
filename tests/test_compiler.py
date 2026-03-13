"""Tests for the compiler: level assignment and execution plan."""

from batch_compiler.compiler import compile_graph
from batch_compiler.graph import TaskGraph
from batch_compiler.task import LLMTask, PyTask


class TestCompiler:
    def test_single_task(self):
        g = TaskGraph()
        g.add(LLMTask(id="t1", prompt="Hi"))
        plan = compile_graph(g)
        assert len(plan.levels) == 1
        assert plan.levels[0].task_ids == ["t1"]
        assert plan.total_tasks == 1

    def test_linear_chain_levels(self):
        g = TaskGraph()
        g.add(LLMTask(id="a", prompt="1"))
        g.add(LLMTask(id="b", prompt="2", deps=["a"]))
        g.add(LLMTask(id="c", prompt="3", deps=["b"]))
        plan = compile_graph(g)
        assert len(plan.levels) == 3
        assert plan.levels[0].task_ids == ["a"]
        assert plan.levels[1].task_ids == ["b"]
        assert plan.levels[2].task_ids == ["c"]

    def test_parallel_tasks_same_level(self):
        g = TaskGraph()
        g.add(LLMTask(id="a", prompt="1"))
        g.add(LLMTask(id="b", prompt="2"))
        g.add(LLMTask(id="c", prompt="3"))
        plan = compile_graph(g)
        assert len(plan.levels) == 1
        assert set(plan.levels[0].task_ids) == {"a", "b", "c"}

    def test_fan_out_two_levels(self):
        g = TaskGraph()
        g.add(LLMTask(id="root", prompt="start"))
        for i in range(3):
            g.add(LLMTask(id=f"child-{i}", prompt=f"c{i}", deps=["root"]))
        plan = compile_graph(g)
        assert len(plan.levels) == 2
        assert plan.levels[0].task_ids == ["root"]
        assert set(plan.levels[1].task_ids) == {"child-0", "child-1", "child-2"}

    def test_diamond_dag(self):
        g = TaskGraph()
        g.add(LLMTask(id="root", prompt="start"))
        g.add(LLMTask(id="left", prompt="l", deps=["root"]))
        g.add(LLMTask(id="right", prompt="r", deps=["root"]))
        g.add(LLMTask(id="join", prompt="j", deps=["left", "right"]))
        plan = compile_graph(g)
        assert len(plan.levels) == 3
        assert plan.levels[0].task_ids == ["root"]
        assert set(plan.levels[1].task_ids) == {"left", "right"}
        assert plan.levels[2].task_ids == ["join"]

    def test_mixed_task_types(self):
        g = TaskGraph()
        g.add(LLMTask(id="llm1", prompt="Hi"))
        g.add(PyTask(id="py1", fn=lambda x: x, deps=["llm1"]))
        g.add(LLMTask(id="llm2", prompt_template="{py1}", deps=["py1"]))
        plan = compile_graph(g)
        assert len(plan.levels) == 3
        assert plan.levels[0].task_ids == ["llm1"]
        assert plan.levels[1].task_ids == ["py1"]
        assert plan.levels[2].task_ids == ["llm2"]
