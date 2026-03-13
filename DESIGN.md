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
form, and it compiles them into a dependency graph, identifies independent LLM
calls, and batches them together when enough are ready.

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

### k-threshold batching

The executor collects **ready** LLM tasks (all deps resolved) and submits them
as an Anthropic batch whenever the count reaches a configurable threshold `k`.
This avoids waiting for an entire level to become ready before dispatching work.

```
Ready queue grows:  [t1, t2, t3, ..., tk]  → submit batch
Batch completes → new tasks become ready → repeat
```

If fewer than `k` tasks remain and no more can become ready, the executor
flushes whatever is in the queue as a final batch.

### PyTask execution

PyTasks run **sequentially** in topological order. When a PyTask becomes ready
(all its deps are resolved), it runs immediately, and its result is added to
the resolved set — potentially unblocking more LLM tasks for the next batch.

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

results = await g.run(batch_threshold=10)
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
              │ • k-threshold   │
              │   batching      │
              │ • sequential    │
              │   PyTask exec   │
              │ • batch submit  │
              │ • poll / await  │
              │ • result map    │
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

### 1. k-threshold batching

Rather than strict level-by-level execution, the executor maintains a **ready
queue** of LLM tasks whose deps are all resolved. Whenever the queue reaches
`k` items (default: 10), those tasks are submitted as a single Anthropic batch.
When no more tasks can become ready and the queue is non-empty, it flushes.

This is more flexible than pure level-based execution: if a PyTask at level 1
completes quickly and unlocks several level-2 LLM tasks, those can be batched
together with remaining level-1 tasks that are still ready.

### 2. Sequential PyTasks

PyTasks run one at a time in dependency order. This keeps the execution model
simple and predictable. Since PyTasks are typically fast glue code (aggregation,
formatting), the overhead of running them serially is negligible compared to
batch API latency.

### 3. Template-based prompt composition

When a task depends on earlier tasks, its prompt is built by substituting
dependency results into a template string:

```python
prompt_template="Classify:\n\n{summarize-3}"
# At execution time, {summarize-3} is replaced with the text output
# of the task with id "summarize-3".
```

Templates use Python's `str.format_map` with a dict of resolved results.
Unresolved references raise `TemplateError` at compile time.

### 4. `foreach` fan-out

The YAML `foreach` key expands a single task definition into N tasks, one per
item. Each gets an auto-generated id like `summarize[0]`, `summarize[1]`, etc.
Dependencies can reference `task[{index}]` for matched fan-out or `task[*]` for
fan-in (depend on all expanded tasks).

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
3. Executor runs using k-threshold strategy:
   a. Initialize ready queue with all level-0 tasks
   b. Loop until all tasks are resolved:
      i.   Run any ready PyTasks sequentially, adding results to resolved set
      ii.  After each PyTask, check if new tasks became ready; add them
      iii. If ready LLM tasks >= k, submit them as a batch
      iv.  If no more PyTasks are ready and LLM queue is non-empty, flush batch
      v.   Poll batch until done; collect results into resolved set
      vi.  Check for newly ready tasks; goto (i)
   c. Return full result map
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
    "jinja2>=3.1",
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

- **Streaming partial results**: start next tasks as soon as their
  specific deps resolve within a batch, without waiting for the full batch.
- **Persistent state / checkpointing**: save intermediate results to disk so
  crashed runs can resume.
- **Web UI**: visualise the DAG and execution progress.
- **Conditional tasks**: skip branches based on runtime predicates.
- **Rate-aware chunking**: split batches with >100k tasks into multiple batches.
- **Parallel PyTasks**: run independent PyTasks concurrently via asyncio.
