# CLAUDE.md

## Project

Task graph batch compiler for the Anthropic Batch API. Compiles DAGs of LLM tasks into leveled execution plans, dispatching via individual async calls or the batch API depending on task count.

## Rules

- Don't make claims that are irrelevant. Stay focused on doing the work. Don't editorialize about difficulty, feasibility, or what "would be needed" — just look it up and do it.
- Don't take shortcuts. Read the docs (platform.claude.com/docs), check the installed SDK version, read actual type definitions. Don't guess, don't str() structured objects, don't assume you know what an API supports.

## Test commands

```
python -m pytest tests/ -v
```

## Structure

- `batch_compiler/` — core library (task, graph, compiler, executor, batch_client)
- `scripts/` — CLI tools (tune_batch_threshold.py)
- `tests/` — pytest tests
