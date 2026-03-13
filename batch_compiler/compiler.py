"""Compiler: transform a TaskGraph into an ExecutionPlan (levels)."""

from __future__ import annotations

from dataclasses import dataclass, field

from .graph import TaskGraph
from .task import Task


@dataclass
class Level:
    """A set of tasks whose deps are all in prior levels."""

    index: int
    task_ids: list[str] = field(default_factory=list)


@dataclass
class ExecutionPlan:
    """An ordered list of levels to execute."""

    levels: list[Level]
    tasks: dict[str, Task]

    @property
    def total_tasks(self) -> int:
        return sum(len(level.task_ids) for level in self.levels)


def compile_graph(graph: TaskGraph) -> ExecutionPlan:
    """Compile a TaskGraph into an ExecutionPlan.

    Validates the DAG, computes levels, and returns an ExecutionPlan.
    Level assignment: level(t) = 0 if no deps, else 1 + max(level(dep) for dep in t.deps).
    """
    graph.validate()
    topo_order = graph.topological_sort()
    tasks = graph.tasks

    # Assign levels
    task_level: dict[str, int] = {}
    for tid in topo_order:
        task = tasks[tid]
        if not task.deps:
            task_level[tid] = 0
        else:
            task_level[tid] = 1 + max(task_level[dep] for dep in task.deps)

    # Group into Level objects
    max_level = max(task_level.values()) if task_level else 0
    levels: list[Level] = []
    for i in range(max_level + 1):
        level_tasks = [tid for tid in topo_order if task_level[tid] == i]
        levels.append(Level(index=i, task_ids=level_tasks))

    return ExecutionPlan(levels=levels, tasks=tasks)
