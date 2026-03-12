# Implementation Plan: Task Graph Batch Compiler

## Steps

### 1. Project scaffolding
- Create `pyproject.toml` with dependencies (anthropic, pyyaml, jinja2, dev tools)
- Create `batch_compiler/` package with `__init__.py`
- Create `tests/` directory

### 2. Core data model (`batch_compiler/task.py`)
- `LLMTask` dataclass: id, deps, prompt/prompt_template, model, max_tokens, system, temperature
- `PyTask` dataclass: id, deps, fn (callable)
- `TaskResult` dataclass: task_id, status, content, error, usage, raw_response

### 3. Task graph (`batch_compiler/graph.py`)
- `TaskGraph` class with `add()`, topological validation, cycle detection
- `from_yaml()` classmethod (delegates to yaml_loader)
- `run()` convenience async method (compiles + executes)

### 4. Compiler (`batch_compiler/compiler.py`)
- `compile(graph) → ExecutionPlan`
- DFS-based cycle detection
- Topological sort
- Level assignment: level = 1 + max(dep levels), roots = 0
- `ExecutionPlan` = list of `Level` (list of tasks per level)

### 5. Template engine (`batch_compiler/template.py`)
- Render `prompt_template` strings by substituting dep results
- Support `{task-id}` placeholders via `str.format_map`
- Jinja2 support for YAML-defined templates
- Validate all references at compile time

### 6. Batch client (`batch_compiler/batch_client.py`)
- Async wrapper over `anthropic.AsyncAnthropic().messages.batches`
- `create_batch(items)` → batch id
- `poll_until_done(batch_id)` with exponential backoff (5s → 60s)
- `get_results(batch_id)` → dict[custom_id, result]
- Retry logic for transient failures

### 7. Executor (`batch_compiler/executor.py`)
- Iterate levels sequentially
- Per level: render templates, build batch items, submit batch, run PyTasks concurrently
- Poll batch, collect results, merge into result map
- Handle errors: mark failed tasks, skip dependents

### 8. YAML loader (`batch_compiler/yaml_loader.py`)
- Parse YAML pipeline definitions
- Expand `foreach` into N concrete tasks with `task[index]` ids
- Resolve `task[{index}]` and `task[*]` dep references
- Wire up defaults (model, max_tokens)

### 9. Error types (`batch_compiler/errors.py`)
- `CycleError`, `TemplateError`, `BatchError`, `DependencyError`

### 10. Tests
- `tests/test_graph.py`: DAG construction, cycle detection
- `tests/test_compiler.py`: level assignment, topo sort
- `tests/test_template.py`: prompt rendering, error cases
- `tests/test_yaml_loader.py`: foreach expansion, dep resolution
- `tests/test_executor.py`: mock batch client, end-to-end level execution

### 11. Commit and push
- Commit all files with descriptive message
- Push to `claude/task-graph-batch-compiler-K3iTD`
