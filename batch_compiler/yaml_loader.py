"""YAML pipeline loader with foreach expansion."""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any

import yaml

from .errors import DependencyError, TemplateError
from .graph import TaskGraph
from .task import LLMTask, PyTask


def load_yaml(path: str, variables: dict[str, Any]) -> TaskGraph:
    """Load a YAML pipeline definition into a TaskGraph.

    Supports:
      - defaults: model, max_tokens applied to all tasks
      - foreach: fan-out a task over items in a variable
      - dep references: task[{index}] for matched fan-out, task[*] for fan-in
      - type: python tasks with fn: module:function
    """
    raw = Path(path).read_text()
    data = yaml.safe_load(raw)

    defaults = data.get("defaults", {})
    task_defs = data.get("tasks", {})

    graph = TaskGraph()

    # First pass: expand all task definitions (to know all task ids)
    expanded: dict[str, list[str]] = {}  # base_name -> list of expanded ids

    for name, defn in task_defs.items():
        foreach_var = defn.get("foreach")
        if foreach_var:
            # Resolve variable reference like "$articles"
            var_name = foreach_var.lstrip("$")
            if var_name not in variables:
                raise TemplateError(
                    f"Task '{name}' foreach references variable '${var_name}' "
                    f"which was not provided"
                )
            items = variables[var_name]
            expanded[name] = [f"{name}[{i}]" for i in range(len(items))]
        else:
            expanded[name] = [name]

    all_task_ids = set()
    for ids in expanded.values():
        all_task_ids.update(ids)

    # Second pass: create task objects
    for name, defn in task_defs.items():
        foreach_var = defn.get("foreach")
        task_type = defn.get("type", "llm")

        if foreach_var:
            var_name = foreach_var.lstrip("$")
            items = variables[var_name]

            for i, item in enumerate(items):
                task_id = f"{name}[{i}]"
                deps = _resolve_deps(defn.get("deps", []), expanded, i)
                _validate_dep_ids(task_id, deps, all_task_ids)

                if task_type == "python":
                    fn = _import_function(defn["fn"])
                    graph.add(PyTask(id=task_id, fn=fn, deps=deps))
                else:
                    prompt = _render_foreach_prompt(defn.get("prompt", ""), item)
                    graph.add(LLMTask(
                        id=task_id,
                        prompt=prompt,
                        model=defn.get("model", defaults.get("model", "claude-sonnet-4-6")),
                        max_tokens=defn.get("max_tokens", defaults.get("max_tokens", 1024)),
                        system=defn.get("system", defaults.get("system")),
                        temperature=defn.get("temperature", defaults.get("temperature")),
                        deps=deps,
                    ))
        else:
            task_id = name
            deps = _resolve_deps(defn.get("deps", []), expanded, None)
            _validate_dep_ids(task_id, deps, all_task_ids)

            if task_type == "python":
                fn = _import_function(defn["fn"])
                graph.add(PyTask(id=task_id, fn=fn, deps=deps))
            else:
                prompt_template = defn.get("prompt")
                graph.add(LLMTask(
                    id=task_id,
                    prompt=prompt_template,
                    model=defn.get("model", defaults.get("model", "claude-sonnet-4-6")),
                    max_tokens=defn.get("max_tokens", defaults.get("max_tokens", 1024)),
                    system=defn.get("system", defaults.get("system")),
                    temperature=defn.get("temperature", defaults.get("temperature")),
                    deps=deps,
                ))

    return graph


def _resolve_deps(
    dep_patterns: list[str],
    expanded: dict[str, list[str]],
    index: int | None,
) -> list[str]:
    """Resolve dependency patterns like 'task[{index}]' and 'task[*]'."""
    resolved: list[str] = []
    for pattern in dep_patterns:
        # task[*] -> all expanded ids for that task
        match_star = re.match(r"^(\w+)\[\*\]$", pattern)
        if match_star:
            base = match_star.group(1)
            if base in expanded:
                resolved.extend(expanded[base])
            else:
                resolved.append(pattern)
            continue

        # task[{index}] -> task[i] using current foreach index
        match_idx = re.match(r"^(\w+)\[\{index\}\]$", pattern)
        if match_idx and index is not None:
            base = match_idx.group(1)
            resolved.append(f"{base}[{index}]")
            continue

        # Literal dep id
        resolved.append(pattern)

    return resolved


def _validate_dep_ids(
    task_id: str, deps: list[str], all_ids: set[str]
) -> None:
    """Raise DependencyError if any dep is not a known task id."""
    for dep in deps:
        if dep not in all_ids:
            raise DependencyError(
                f"Task '{task_id}' depends on '{dep}' which does not exist"
            )


def _render_foreach_prompt(prompt: str, item: Any) -> str:
    """Replace {{ item }} in a prompt with the foreach value."""
    return prompt.replace("{{ item }}", str(item))


def _import_function(fn_ref: str) -> Any:
    """Import a function from 'module:function_name' notation."""
    module_path, _, func_name = fn_ref.rpartition(":")
    if not module_path:
        raise TemplateError(f"Invalid function reference '{fn_ref}', expected 'module:function'")
    module = importlib.import_module(module_path)
    return getattr(module, func_name)
