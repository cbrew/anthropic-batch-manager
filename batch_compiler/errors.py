"""Error types for the batch compiler."""


class BatchCompilerError(Exception):
    """Base error for all batch compiler errors."""


class CycleError(BatchCompilerError):
    """The task graph contains a cycle."""


class DuplicateTaskError(BatchCompilerError):
    """A task with the same id already exists in the graph."""


class DependencyError(BatchCompilerError):
    """A task references a dependency that does not exist in the graph."""


class TemplateError(BatchCompilerError):
    """A prompt template references an unknown or unresolved dependency."""


class BatchError(BatchCompilerError):
    """An error occurred while communicating with the Anthropic Batch API."""
