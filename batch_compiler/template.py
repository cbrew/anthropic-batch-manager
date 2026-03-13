"""Prompt template rendering with dependency substitution."""

from __future__ import annotations

import re

from .errors import TemplateError


def render_template(template: str, resolved: dict[str, str]) -> str:
    """Render a prompt template by substituting {task-id} placeholders.

    Args:
        template: A string with {task-id} placeholders.
        resolved: Map of task_id -> output text for resolved dependencies.

    Returns:
        The rendered prompt string.

    Raises:
        TemplateError: If a placeholder references an unresolved task.
    """
    # Find all {placeholder} references in the template
    placeholders = re.findall(r"\{([^}]+)\}", template)

    for ref in placeholders:
        if ref not in resolved:
            raise TemplateError(
                f"Template references '{ref}' but it is not in resolved results. "
                f"Available: {sorted(resolved.keys())}"
            )

    # Manual substitution to handle keys with brackets/hyphens that
    # str.format_map would misparse (e.g. {summarize[0]} is treated as indexing)
    result = template
    for ref in placeholders:
        result = result.replace("{" + ref + "}", resolved[ref])
    return result


def validate_template_refs(template: str, available_ids: set[str]) -> list[str]:
    """Check that all placeholders in a template refer to known task ids.

    Returns list of referenced task ids.
    Raises TemplateError if any reference is unknown.
    """
    refs = re.findall(r"\{([^}]+)\}", template)
    unknown = [r for r in refs if r not in available_ids]
    if unknown:
        raise TemplateError(
            f"Template references unknown task ids: {unknown}. "
            f"Known ids: {sorted(available_ids)}"
        )
    return refs
