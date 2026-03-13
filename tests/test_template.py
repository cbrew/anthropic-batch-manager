"""Tests for prompt template rendering."""

import pytest

from batch_compiler.errors import TemplateError
from batch_compiler.template import render_template, validate_template_refs


class TestRenderTemplate:
    def test_simple_substitution(self):
        result = render_template(
            "Classify: {summary}", {"summary": "This is about AI."}
        )
        assert result == "Classify: This is about AI."

    def test_multiple_refs(self):
        result = render_template(
            "Compare {a} with {b}",
            {"a": "apples", "b": "oranges"},
        )
        assert result == "Compare apples with oranges"

    def test_hyphenated_id(self):
        result = render_template(
            "Result: {task-1}", {"task-1": "done"}
        )
        assert result == "Result: done"

    def test_bracketed_id(self):
        result = render_template(
            "Result: {summarize[0]}", {"summarize[0]": "summary text"}
        )
        assert result == "Result: summary text"

    def test_missing_ref_raises(self):
        with pytest.raises(TemplateError, match="not in resolved"):
            render_template("Use {missing}", {})

    def test_no_placeholders(self):
        result = render_template("Plain prompt", {})
        assert result == "Plain prompt"


class TestValidateTemplateRefs:
    def test_valid_refs(self):
        refs = validate_template_refs("{a} and {b}", {"a", "b", "c"})
        assert set(refs) == {"a", "b"}

    def test_unknown_ref_raises(self):
        with pytest.raises(TemplateError, match="unknown task ids"):
            validate_template_refs("{a} and {missing}", {"a"})

    def test_no_refs(self):
        refs = validate_template_refs("no refs here", {"a"})
        assert refs == []
