# CLAUDE.md

## Project

Task graph batch compiler for the Anthropic Batch API. Compiles DAGs of LLM tasks into leveled execution plans, dispatching via individual async calls or the batch API depending on task count.

## Rules

- Never claim something is impossible, requires a large change, or "would need X" without first checking the SDK, docs, or codebase. If you don't know, look it up. Do not speculate about API capabilities.
- When working with the Anthropic SDK: check the installed version, read the actual type definitions or official docs. Do not guess at what parameters exist or what the API supports.
- Do not str() structured objects (errors, responses, SDK types) to flatten them into strings. Extract the specific fields you need.
- Check the Anthropic docs (platform.claude.com/docs) for current API features before making claims about what is or isn't supported.

## Test commands

```
python -m pytest tests/ -v
```

## Structure

- `batch_compiler/` — core library (task, graph, compiler, executor, batch_client)
- `scripts/` — CLI tools (tune_batch_threshold.py)
- `tests/` — pytest tests
