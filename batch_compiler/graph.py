"""TaskGraph: DAG of tasks with validation and convenience methods."""

from __future__ import annotations

from typing import Any

from .errors import CycleError, DependencyError, DuplicateTaskError
from .task import LLMTask, PyTask, Task


class TaskGraph:
    """A directed acyclic graph of LLMTask and PyTask nodes."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def add(self, task: Task) -> TaskGraph:
        """Add a task to the graph. Returns self for chaining."""
        if task.id in self._tasks:
            raise DuplicateTaskError(f"Task '{task.id}' already exists")
        self._tasks[task.id] = task
        return self

    @property
    def tasks(self) -> dict[str, Task]:
        return dict(self._tasks)

    @property
    def task_ids(self) -> set[str]:
        return set(self._tasks.keys())

    def __len__(self) -> int:
        return len(self._tasks)

    def validate(self) -> None:
        """Validate the graph: check deps exist and no cycles.

        Raises:
            DependencyError: If a task references a non-existent dep.
            CycleError: If the graph contains a cycle.
        """
        # Check all deps reference existing tasks
        for task in self._tasks.values():
            for dep_id in task.deps:
                if dep_id not in self._tasks:
                    raise DependencyError(
                        f"Task '{task.id}' depends on '{dep_id}' "
                        f"which does not exist in the graph"
                    )

        # Detect cycles via DFS
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {tid: WHITE for tid in self._tasks}
        path: list[str] = []

        def dfs(node_id: str) -> None:
            color[node_id] = GRAY
            path.append(node_id)
            for dep_id in self._tasks[node_id].deps:
                if color[dep_id] == GRAY:
                    cycle_start = path.index(dep_id)
                    cycle = path[cycle_start:] + [dep_id]
                    raise CycleError(
                        f"Cycle detected: {' -> '.join(cycle)}"
                    )
                if color[dep_id] == WHITE:
                    dfs(dep_id)
            path.pop()
            color[node_id] = BLACK

        for tid in self._tasks:
            if color[tid] == WHITE:
                dfs(tid)

    def topological_sort(self) -> list[str]:
        """Return task ids in topological order (dependencies first).

        Raises:
            DependencyError: If deps are missing.
            CycleError: If cycles exist.
        """
        self.validate()

        in_degree: dict[str, int] = {tid: 0 for tid in self._tasks}
        dependents: dict[str, list[str]] = {tid: [] for tid in self._tasks}

        for task in self._tasks.values():
            for dep_id in task.deps:
                in_degree[task.id] += 1
                dependents[dep_id].append(task.id)

        # Kahn's algorithm
        queue = sorted(tid for tid, deg in in_degree.items() if deg == 0)
        result: list[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for dependent in sorted(dependents[node]):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        return result

    @classmethod
    def from_yaml(
        cls, path: str, variables: dict[str, Any] | None = None
    ) -> TaskGraph:
        """Load a TaskGraph from a YAML pipeline definition."""
        from .yaml_loader import load_yaml

        return load_yaml(path, variables or {})

    async def run(
        self,
        client: Any | None = None,
        max_batch_size: int = 100_000,
    ) -> dict[str, Any]:
        """Compile and execute the graph.

        Args:
            client: Optional anthropic.AsyncAnthropic client instance.
            max_batch_size: Max requests per Anthropic batch (default 100k).

        Returns:
            Dict mapping task_id -> TaskResult.
        """
        from .compiler import compile_graph
        from .executor import Executor

        plan = compile_graph(self)
        executor = Executor(
            plan=plan,
            client=client,
            max_batch_size=max_batch_size,
        )
        return await executor.run()
