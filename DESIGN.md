# Task Graph Batch Compiler — Design

## Problem

Users naturally think about LLM-powered workflows as sequential steps:

```
1. Summarize each of these 50 articles
2. Classify each summary by topic
3. Generate a report from the classified summaries
```

But step 1 contains 50 **independent** calls, and step 2 contains 50 more that
each depend only on one result from step 1 — not on each other. Naively running
them serially wastes time and money. The Anthropic Batch API offers a **50%
price reduction** and high throughput, but requires the caller to manually
assemble flat lists of requests.

**The task graph compiler bridges this gap**: you write tasks in natural serial
form, and it compiles them into a dependency graph, groups independent nodes
into batch API calls, and executes the graph level-by-level.

---

## Core Concepts

### Task

A single unit of work that produces a result. Two flavours:

| Kind        | Description                                          |
|-------------|------------------------------------------------------|
| `LLMTask`   | An Anthropic Messages API call (prompt + params)     |
| `PyTask`    | An arbitrary Python callable (for glue / transforms) |

Every task has:

- **`id`** — unique string (auto-generated or user-supplied)
- **`deps`** — list of task ids this task depends on
- **`template`** — a prompt template (LLMTask) or callable (PyTask) that
  receives the resolved results of its deps

### TaskGraph

A directed acyclic graph (DAG) of tasks. The compiler validates it is acyclic,
resolves dependency order, and partitions nodes into **levels** — sets of tasks
whose dependencies are all satisfied by prior levels.

### Execution plan

```
Level 0:  [task_a, task_b, task_c]   ← all independent → one Batch API call
Level 1:  [task_d, task_e]           ← depend on level-0 → second Batch call
Level 2:  [task_f]                   ← depends on level-1 → third Batch call
```

Within each level:
- All `LLMTask` nodes are submitted as a **single Anthropic batch**.
- All `PyTask` nodes run concurrently via `asyncio`.
- The level completes when both the batch and all PyTasks finish.

---

## User-Facing API

### 1. Imperative builder

```python
from batch_compiler import TaskGraph, LLMTask, PyTask

g = TaskGraph()

# Fan-out: 50 independent summarisation tasks
for i, article in enumerate(articles):
    g.add(LLMTask(
        id=f"summarize-{i}",
        prompt=f"Summarize this article:\n\n{article}",
        model="claude-sonnet-4-6",
        max_tokens=300,
    ))

# Fan-in per item: classify each summary (depends on its summarize task)
for i in range(len(articles)):
    g.add(LLMTask(
        id=f"classify-{i}",
        deps=[f"summarize-{i}"],
        prompt_template="Classify this summary into a topic:\n\n{summarize-" + str(i) + "}",
        model="claude-haiku-4-5-20251001",
        max_tokens=50,
    ))

# Final aggregation (Python glue)
g.add(PyTask(
    id="report",
    deps=[f"classify-{i}" for i in range(len(articles))],
    fn=build_report,   # user-defined function
))

results = await g.run()
print(results["report"])
```

### 2. Declarative YAML

```yaml
# pipeline.yaml
defaults:
  model: claude-sonnet-4-6
  max_tokens: 300

tasks:
  summarize:
    foreach: $articles            # fan-out over input variable
    prompt: |
      Summarize this article:
      {{ item }}

  classify:
    foreach: $articles
    deps: ["summarize[{index}]"]  # each depends on matching summarize
    model: claude-haiku-4-5-20251001
    max_tokens: 50
    prompt: |
      Classify this summary into a topic:
      {{ deps.summarize }}

  report:
    type: python
    deps: ["classify[*]"]         # depends on ALL classify tasks
    fn: my_module:build_report
```

```python
from batch_compiler import TaskGraph

g = TaskGraph.from_yaml("pipeline.yaml", variables={"articles": articles})
results = await g.run()
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    User API layer                    │
│  TaskGraph  ·  LLMTask  ·  PyTask  ·  YAML loader   │
└──────────────────────┬──────────────────────────────┘
                       │
              ┌────────▼────────┐
              │    Compiler     │
              │ ────────────── │
              │ • validate DAG  │
              │ • topo-sort     │
              │ • partition into│
              │   levels        │
              │ • template      │
              │   resolution    │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │    Executor     │
              │ ────────────── │
              │ • per-level     │
              │   dispatch      │
              │ • batch submit  │
              │ • poll / await  │
              │ • result map    │
              │ • retry logic   │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  Batch Client   │
              │ ────────────── │
              │ • thin wrapper  │
              │   over anthropic│
              │   SDK batches   │
              │ • create, poll, │
              │   retrieve      │
              └─────────────────┘
```

### Module layout

```
batch_compiler/
├── __init__.py          # public re-exports
├── task.py              # LLMTask, PyTask, TaskResult dataclasses
├── graph.py             # TaskGraph: add, validate, from_yaml
├── compiler.py          # compile(graph) → ExecutionPlan (levels)
├── executor.py          # run an ExecutionPlan using the batch client
├── batch_client.py      # async wrapper around anthropic SDK batches
├── template.py          # prompt template rendering (dep substitution)
├── yaml_loader.py       # YAML → TaskGraph, foreach expansion
└── errors.py            # CycleError, TemplateError, BatchError, etc.
```

---

## Key Design Decisions

### 1. Level-based execution (not streaming)

The graph executes **level by level**. All tasks in a level are submitted as one
batch, and we wait for the entire batch to complete before advancing. This is
simpler than streaming partial results and matches the Batch API's semantics
(results arrive when the batch ends).

### 2. Template-based prompt composition

When a task depends on earlier tasks, its prompt is built by substituting
dependency results into a template string:

```python
prompt_template="Classify:\n\n{summarize-3}"
# At execution time, {summarize-3} is replaced with the text output
# of the task with id "summarize-3".
```

Templates use Python's `str.format_map` with a dict of resolved results.
Unresolved references raise `TemplateError` at compile time.

### 3. `foreach` fan-out

The YAML `foreach` key expands a single task definition into N tasks, one per
item. Each gets an auto-generated id like `summarize[0]`, `summarize[1]`, etc.
Dependencies can reference `task[{index}]` for matched fan-out or `task[*]` for
fan-in (depend on all expanded tasks).

### 4. Mixed LLM + Python tasks

Not every step needs an LLM. `PyTask` lets users insert arbitrary transforms
(aggregation, filtering, formatting) into the graph. PyTasks within a level
run as `asyncio` tasks concurrently with the batch API call for that level.

### 5. Retry and error handling

- **Batch-level**: if the batch creation fails, retry with exponential backoff
  (up to 3 attempts).
- **Request-level**: individual requests that error within a batch are recorded
  in the result. Dependent tasks that cannot resolve their deps are marked
  `skipped` rather than crashing the whole graph.
- **Polling**: exponential backoff starting at 5s, capping at 60s.

### 6. Result storage

`TaskResult` is a dataclass holding:

```python
@dataclass
class TaskResult:
    task_id: str
    status: Literal["succeeded", "errored", "skipped"]
    content: str | None          # text output for succeeded
    error: str | None            # error message for errored/skipped
    usage: Usage | None          # token counts
    raw_response: dict | None    # full API response (optional)
```

The executor returns `dict[str, TaskResult]` keyed by task id.

---

## Execution Flow (detailed)

```
1. User builds TaskGraph (imperative or YAML)
2. Compiler:
   a. Validate DAG (detect cycles via DFS)
   b. Topological sort
   c. Assign each node a level = 1 + max(level of deps), roots = level 0
   d. Return ExecutionPlan = list[Level], where Level = list[Task]
3. Executor iterates levels 0 → N:
   a. For each LLMTask in the level:
      - Render prompt template using resolved results from prior levels
      - Build batch request item {custom_id: task.id, params: {...}}
   b. Submit all LLMTask items as one Anthropic batch
   c. Run all PyTask callables concurrently (passing resolved deps)
   d. Poll batch until processing_status == "ended"
   e. Retrieve batch results, map custom_id → TaskResult
   f. Merge PyTask results into the result map
   g. Advance to next level
4. Return full result map to user
```

---

## Dependencies

```toml
[project]
name = "batch-compiler"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.40.0",
    "pyyaml>=6.0",
    "jinja2>=3.1",          # for richer templates in YAML mode
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "ruff>=0.8",
    "mypy>=1.13",
]
```

---

## Example: end-to-end batch savings

| Scenario | Serial API calls | Task Graph Compiler |
|----------|-----------------|---------------------|
| 50 summaries + 50 classifications + 1 report | 101 API calls, full price | 2 batch calls (50% off) + 1 PyTask |
| Wall-clock time | ~101 × avg latency | 2 × batch latency (~minutes) |
| Cost | 100% | ~50% |

---

## Future Extensions (out of scope for v1)

- **Streaming partial results**: start level N+1 tasks as soon as their
  specific deps resolve, without waiting for the full batch.
- **Persistent state / checkpointing**: save intermediate results to disk so
  crashed runs can resume.
- **Web UI**: visualise the DAG and execution progress.
- **Conditional tasks**: skip branches based on runtime predicates.
- **Rate-aware chunking**: split levels with >100k tasks into multiple batches.
