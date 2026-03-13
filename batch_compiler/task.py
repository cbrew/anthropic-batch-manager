"""Core task and result data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal


@dataclass
class LLMTask:
    """A task that makes an Anthropic Messages API call."""

    id: str
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024
    prompt: str | None = None
    prompt_template: str | None = None
    system: str | None = None
    temperature: float | None = None
    deps: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.prompt is None and self.prompt_template is None:
            raise ValueError(
                f"LLMTask '{self.id}': must provide either prompt or prompt_template"
            )
        if self.prompt is not None and self.prompt_template is not None:
            raise ValueError(
                f"LLMTask '{self.id}': provide prompt or prompt_template, not both"
            )


@dataclass
class PyTask:
    """A task that runs a Python callable."""

    id: str
    fn: Callable[..., Any]
    deps: list[str] = field(default_factory=list)


# Union type for any task
Task = LLMTask | PyTask


@dataclass
class TaskResult:
    """The result of executing a single task."""

    task_id: str
    status: Literal["succeeded", "errored", "skipped"]
    content: str | None = None
    error: str | None = None
    usage: dict[str, int] | None = None
    raw_response: dict[str, Any] | None = None
