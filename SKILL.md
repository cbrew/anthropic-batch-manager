# batch-compiler: Task Graph Batch Compiler for Anthropic API

## What This Is

`batch-compiler` is a Python library that compiles DAGs of LLM tasks into optimized execution plans. You define tasks and their dependencies; the compiler partitions them into levels, runs independent tasks in parallel, and automatically routes through either concurrent individual API calls (low latency) or the Anthropic Batch API (50% cost reduction) based on a configurable threshold.

**When to use this skill**: When the user's code imports `batch_compiler`, references `TaskGraph`, `LLMTask`, `PyTask`, or discusses Anthropic Batch API workflows, pipeline orchestration of LLM calls, or fan-out/fan-in patterns over Claude models.

---

## Package Layout

```
batch_compiler/
├── __init__.py          # Public re-exports (everything below)
├── task.py              # LLMTask, PyTask, TaskResult dataclasses
├── graph.py             # TaskGraph: DAG container, validation, execution entry point
├── compiler.py          # compile_graph() → ExecutionPlan with Level objects
├── executor.py          # Runs an ExecutionPlan level-by-level
├── batch_client.py      # Async wrapper: create, poll, retrieve Anthropic batches
├── template.py          # {task-id} placeholder substitution in prompts
├── yaml_loader.py       # YAML → TaskGraph, with foreach expansion
└── errors.py            # Exception hierarchy
```

---

## Public API Reference

### Imports

```python
from batch_compiler import (
    TaskGraph,          # DAG container
    LLMTask,            # Anthropic API call task
    PyTask,             # Python callable task
    TaskResult,         # Result of any task execution
    compile_graph,      # TaskGraph → ExecutionPlan
    ExecutionPlan,      # Ordered list of Levels
    Level,              # Group of tasks at same depth
    # Errors
    BatchCompilerError, # Base exception
    CycleError,         # Graph has a cycle
    DuplicateTaskError, # Same task id added twice
    DependencyError,    # Task references nonexistent dep
    TemplateError,      # Prompt template has unresolved placeholder
    BatchError,         # Anthropic Batch API communication failure
)
```

### LLMTask

A task that makes one Anthropic Messages API call.

```python
@dataclass
class LLMTask:
    id: str                                     # Unique identifier
    model: str = "claude-sonnet-4-6"           # Model name
    max_tokens: int = 1024                      # Max output tokens
    prompt: str | None = None                   # Literal prompt text
    prompt_template: str | None = None          # Template with {dep-id} placeholders
    system: str | None = None                   # System message
    temperature: float | None = None            # Sampling temperature
    deps: list[str] = field(default_factory=list)  # Dependency task ids
```

**Constraints:**
- Exactly one of `prompt` or `prompt_template` must be provided (enforced in `__post_init__`).
- `prompt` is used for tasks with no upstream text dependency (or when you inline the text yourself).
- `prompt_template` is used when the prompt needs to include output from dependency tasks. Placeholders like `{task-id}` are resolved at runtime with the text output of the named task.

### PyTask

A task that runs a synchronous Python callable.

```python
@dataclass
class PyTask:
    id: str                                     # Unique identifier
    fn: Callable[..., Any]                     # Called with fn(dep_results: dict[str, str])
    deps: list[str] = field(default_factory=list)
```

**The `fn` signature**: receives a single `dict[str, str]` argument mapping each dependency task id to its text output. The return value is converted to `str` and stored as the task's content.

### TaskResult

Returned for every task after execution.

```python
@dataclass
class TaskResult:
    task_id: str                                          # Which task
    status: Literal["succeeded", "errored", "skipped"]    # Outcome
    content: str | None = None        # Text output (succeeded only)
    error: str | None = None          # Error message (errored/skipped)
    usage: dict[str, int] | None = None   # {"input_tokens": N, "output_tokens": N}
    raw_response: dict[str, Any] | None = None  # Full API response (optional)
```

**Status meanings:**
- `"succeeded"` — task ran and produced content.
- `"errored"` — task ran but failed (API error, template error, Python exception).
- `"skipped"` — task was not attempted because a dependency errored or was skipped.

### TaskGraph

The main user-facing class. A mutable DAG of tasks.

```python
class TaskGraph:
    def __init__(self) -> None

    def add(self, task: LLMTask | PyTask) -> TaskGraph
        # Add a task. Returns self for chaining. Raises DuplicateTaskError.

    def validate(self) -> None
        # Check all deps exist and graph is acyclic.
        # Raises DependencyError or CycleError.

    def topological_sort(self) -> list[str]
        # Returns task ids in dependency order (calls validate first).

    @property
    def tasks(self) -> dict[str, LLMTask | PyTask]   # All tasks by id
    @property
    def task_ids(self) -> set[str]                     # Set of all ids
    def __len__(self) -> int                           # Number of tasks

    @classmethod
    def from_yaml(
        cls,
        path: str,
        variables: dict[str, Any] | None = None,
    ) -> TaskGraph
        # Load from a YAML pipeline file. See YAML section below.

    async def run(
        self,
        client: anthropic.AsyncAnthropic | None = None,  # Uses default if None
        max_batch_size: int = 100_000,                    # Anthropic API limit
        batch_threshold: int = 10,                        # Switch point
    ) -> dict[str, TaskResult]
        # Compile + execute the graph. Returns results for all tasks.
```

**`batch_threshold` parameter** (critical for understanding dispatch behavior):
- If a level has **fewer than** `batch_threshold` LLM tasks → concurrent individual `messages.create()` calls.
- If a level has `batch_threshold` **or more** LLM tasks → Anthropic Batch API, auto-chunked into groups of `max_batch_size`.

### compile_graph

```python
def compile_graph(graph: TaskGraph) -> ExecutionPlan
```

Validates the DAG, topologically sorts it, and assigns each task a level:
- Level 0: tasks with no dependencies.
- Level N: `1 + max(level of each dependency)`.

Returns an `ExecutionPlan` containing `Level` objects, each with a list of task ids.

---

## Architecture: How Execution Works

```
User builds TaskGraph (imperative or YAML)
         │
         ▼
┌─────────────────────┐
│  compile_graph()    │  Validate DAG, topological sort, assign levels
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Executor.run()     │  Iterate levels 0 → N
│                     │
│  Per level:         │
│  1. Skip tasks with │  If any dep errored/skipped → mark "skipped"
│     failed deps     │
│  2. Run PyTasks     │  Sequentially, in dependency order
│     sequentially    │
│  3. Dispatch LLM    │  Resolve prompt templates using prior outputs
│     tasks           │  then:
│     < threshold →   │    concurrent messages.create() calls
│     ≥ threshold →   │    Batch API (submit → poll → retrieve)
└─────────┬───────────┘
          ▼
dict[str, TaskResult]    Results for every task in the graph
```

### Batch API Flow (inside `_run_llm_batched`)

1. Chunk tasks into groups of ≤ `max_batch_size` (default 100k).
2. For each chunk: `submit_batch()` → `poll_until_done()` → `get_results()`.
3. `submit_batch`: retries up to 3 times with exponential backoff (2s, 4s, 8s).
4. `poll_until_done`: polls status with backoff from 5s → capped at 60s (factor 1.5x).
5. `get_results`: async iteration over batch results; maps each to a `TaskResult`.

### Prompt Template Resolution

Templates use `{task-id}` placeholders (not Python format strings). At runtime, each placeholder is replaced with the text output of the referenced task.

```python
# These are valid placeholders:
"{summarize-3}"       # hyphenated ids
"{classify[0]}"       # bracket ids from foreach expansion
"{my_task}"           # underscore ids
```

Resolution happens via regex + manual string replacement (not `str.format_map`), which correctly handles brackets and hyphens in task ids.

---

## Usage Patterns

### Pattern 1: Simple Fan-Out

All tasks are independent; they run in parallel in a single level.

```python
from batch_compiler import TaskGraph, LLMTask

g = TaskGraph()
for i, article in enumerate(articles):
    g.add(LLMTask(
        id=f"summarize-{i}",
        prompt=f"Summarize this article:\n\n{article}",
        model="claude-sonnet-4-6",
        max_tokens=300,
    ))

results = await g.run()
for i in range(len(articles)):
    print(results[f"summarize-{i}"].content)
```

### Pattern 2: Fan-Out → Fan-In with Templates

Each classification depends on exactly one summary. A final PyTask aggregates all classifications.

```python
from batch_compiler import TaskGraph, LLMTask, PyTask

g = TaskGraph()

# Level 0: fan-out
for i, article in enumerate(articles):
    g.add(LLMTask(id=f"sum-{i}", prompt=f"Summarize:\n\n{article}"))

# Level 1: fan-out with deps
for i in range(len(articles)):
    g.add(LLMTask(
        id=f"classify-{i}",
        deps=[f"sum-{i}"],
        prompt_template="Classify this summary into a topic:\n\n{sum-" + str(i) + "}",
        model="claude-haiku-4-5-20251001",
        max_tokens=50,
    ))

# Level 2: fan-in (PyTask)
def build_report(dep_results: dict[str, str]) -> str:
    return "\n".join(f"- {v}" for v in dep_results.values())

g.add(PyTask(
    id="report",
    deps=[f"classify-{i}" for i in range(len(articles))],
    fn=build_report,
))

results = await g.run()
print(results["report"].content)
```

### Pattern 3: Linear Chain

Tasks execute one after another, each depending on the previous.

```python
g = TaskGraph()
g.add(LLMTask(id="draft", prompt="Write a draft about AI safety."))
g.add(LLMTask(
    id="critique",
    deps=["draft"],
    prompt_template="Critique this draft:\n\n{draft}",
))
g.add(LLMTask(
    id="revise",
    deps=["draft", "critique"],
    prompt_template="Revise this draft:\n\n{draft}\n\nBased on this critique:\n\n{critique}",
))
results = await g.run()
```

### Pattern 4: YAML Pipeline

```yaml
# pipeline.yaml
defaults:
  model: claude-sonnet-4-6
  max_tokens: 300
  system: "You are a helpful research assistant."

tasks:
  summarize:
    foreach: $articles          # Expands to summarize[0], summarize[1], ...
    prompt: |
      Summarize this article:
      {{ item }}

  classify:
    foreach: $articles
    deps: ["summarize[{index}]"]   # classify[i] depends on summarize[i]
    model: claude-haiku-4-5-20251001
    max_tokens: 50
    prompt: |
      Classify this summary into a topic:
      {{ item }}

  report:
    type: python
    deps: ["classify[*]"]          # Depends on ALL classify tasks
    fn: my_module:build_report     # module:function notation
```

```python
g = TaskGraph.from_yaml("pipeline.yaml", variables={"articles": articles})
results = await g.run()
```

**YAML `foreach` mechanics:**
- `foreach: $variable_name` — iterates over `variables["variable_name"]`.
- Creates tasks: `task_name[0]`, `task_name[1]`, etc.
- `{{ item }}` in prompt is replaced with the current foreach value.
- Dep `task[{index}]` resolves to `task[i]` matching the current iteration index.
- Dep `task[*]` expands to all expanded ids: `task[0]`, `task[1]`, etc.
- `fn: module.path:function_name` imports via `importlib.import_module`.

### Pattern 5: Custom Client and Threshold

```python
import anthropic

client = anthropic.AsyncAnthropic(api_key="sk-...")
results = await g.run(
    client=client,
    batch_threshold=50,    # Only batch when 50+ LLM tasks in a level
    max_batch_size=10_000, # Smaller chunks
)
```

---

## Error Handling

### Exception Hierarchy

```
BatchCompilerError (base)
├── CycleError            # Graph contains a cycle
├── DuplicateTaskError    # Task id already exists in graph
├── DependencyError       # Task references nonexistent dependency
├── TemplateError         # Prompt template has unresolved placeholder
└── BatchError            # Anthropic Batch API failure (after retries)
```

### Error Propagation in Execution

- If a task **errors** (API failure, template error, Python exception), its `TaskResult.status` is `"errored"`.
- Any downstream task whose dependency errored or was skipped is marked `"skipped"` with `error="Dependency failed"`. The skipped task is **not executed**.
- This cascades: if A errors, B (depends on A) is skipped, C (depends on B) is also skipped.
- Tasks in the same level that do **not** depend on the failed task continue normally.

### Retry Behavior

- **Batch creation**: up to 3 attempts with exponential backoff (2s, 4s, 8s). Raises `BatchError` on final failure.
- **Batch polling**: exponential backoff from 5s to 60s max (factor 1.5x). No retry limit — polls until `processing_status == "ended"`.
- **Individual API calls**: no automatic retry. A failed call marks the task `"errored"`.

---

## Testing Guide

### Mocking the Individual Call Path

```python
from unittest.mock import AsyncMock, MagicMock

mock_client = MagicMock()
mock_client.messages = MagicMock()
mock_client.messages.create = AsyncMock(side_effect=mock_create_fn)

executor = Executor(plan=plan, client=mock_client)
results = await executor.run()
```

The `messages.create` mock receives the same `**kwargs` as the real API: `model`, `max_tokens`, `messages`, optionally `system` and `temperature`.

### Mocking the Batch Path

```python
mock_batch = MagicMock()
mock_batch.submit_batch = AsyncMock(return_value="batch-123")
mock_batch.poll_until_done = AsyncMock()
mock_batch.get_results = AsyncMock(return_value={
    "task_id": TaskResult(task_id="task_id", status="succeeded", content="output")
})

executor = Executor(plan=plan, batch_threshold=1)  # Force batch path
executor._batch_client = mock_batch  # Inject mock
```

### Forcing a Specific Dispatch Path

- **Force individual**: set `batch_threshold` higher than the number of LLM tasks in any level.
- **Force batch**: set `batch_threshold=1`.

---

## Batch Threshold Tuning Results

Empirical measurements using `scripts/tune_batch_threshold.py` with `claude-haiku-4-5-20251001` (one-word responses, minimal payload). These results inform the choice of `batch_threshold`.

### Small Scale (3–50 tasks)

| Tasks | Individual (s) | Batch (s) | Winner     |
|------:|---------------:|----------:|:-----------|
|     3 |           1.06 |    222.79 | individual |
|     5 |           0.83 |    282.93 | individual |
|     8 |           0.87 |    105.46 | individual |
|    10 |           1.18 |    105.45 | individual |
|    15 |           1.08 |    162.54 | individual |
|    20 |           1.43 |    105.42 | individual |
|    50 |           2.18 |    105.54 | individual |

### Key Observations

1. **Batch API has a fixed floor of ~100–280 seconds** regardless of payload size. This is server-side scheduling latency, not proportional to request count.
2. **Individual concurrent calls scale near-linearly** and remain under 3 seconds even at 50 tasks.
3. **The crossover point** where batch becomes faster than individual calls is well above 50 tasks — the cost savings (50% discount) must justify the latency penalty.
4. **Default `batch_threshold=10` is too aggressive for latency-sensitive workloads.** A threshold of 50–100+ is more appropriate if wall-clock time matters.
5. **For pure cost optimization** (where latency is acceptable), batch is always cheaper at any task count due to the 50% price reduction.

### Recommendation

- **Latency-sensitive**: set `batch_threshold` to 100+ (or higher).
- **Cost-sensitive, latency-tolerant**: set `batch_threshold` to 10 (default) or even lower.
- **Hybrid**: the default of 10 provides a reasonable middle ground for most pipelines where levels have varying task counts.

---

## Configuration Reference

| Parameter | Default | Where Set | Description |
|-----------|---------|-----------|-------------|
| `batch_threshold` | `10` | `TaskGraph.run()` / `Executor.__init__()` | LLM tasks per level below this count use individual calls; at or above use Batch API |
| `max_batch_size` | `100,000` | `TaskGraph.run()` / `Executor.__init__()` | Maximum requests per Anthropic batch (API hard limit) |
| `poll_initial_delay` | `5.0` s | `BatchClient.__init__()` | Initial polling interval |
| `poll_max_delay` | `60.0` s | `BatchClient.__init__()` | Maximum polling interval |
| `max_retries` | `3` | `BatchClient.__init__()` | Batch creation retry attempts |

---

## Dependencies

- Python ≥ 3.11
- `anthropic >= 0.40.0`
- `pyyaml >= 6.0`
- `jinja2 >= 3.1`

Dev: `pytest >= 8.0`, `pytest-asyncio >= 0.24`, `ruff >= 0.8`, `mypy >= 1.13`

---

## Common Tasks for Code Assistants

### Adding a new task type

1. Define a new dataclass in `task.py` alongside `LLMTask` and `PyTask`.
2. Update the `Task` union type: `Task = LLMTask | PyTask | NewTask`.
3. Handle it in `executor.py` in the level dispatch logic (`run()` method).
4. If it needs YAML support, add parsing in `yaml_loader.py`.

### Modifying dispatch behavior

The dispatch decision lives in `executor.py:_run_llm_level()` (line ~152):
```python
if len(prepared) < self._batch_threshold:
    await self._run_llm_individual(prepared)
else:
    await self._run_llm_batched(prepared)
```

### Adding concurrency limits to individual calls

Currently `_run_llm_individual` uses bare `asyncio.gather()` with no semaphore. To add a concurrency limit, wrap the gather with `asyncio.Semaphore`:

```python
sem = asyncio.Semaphore(50)
async def limited_call(tid, params):
    async with sem:
        await call_one(tid, params)
await asyncio.gather(*(limited_call(tid, params) for tid, params in prepared))
```

### Adding a new prompt variable to YAML

Prompt variables come from the `variables` dict passed to `TaskGraph.from_yaml()`. The YAML references them as `$variable_name` in `foreach` directives. To add new variable usage patterns, modify `yaml_loader.py`.
