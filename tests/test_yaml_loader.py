"""Tests for YAML pipeline loading."""

import textwrap
import tempfile
from pathlib import Path

import pytest

from batch_compiler.errors import DependencyError, TemplateError
from batch_compiler.task import LLMTask, PyTask
from batch_compiler.yaml_loader import load_yaml


def _write_yaml(content: str) -> str:
    """Write YAML content to a temp file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    f.write(textwrap.dedent(content))
    f.close()
    return f.name


class TestYAMLLoader:
    def test_simple_tasks(self):
        path = _write_yaml("""\
            defaults:
              model: claude-sonnet-4-6
              max_tokens: 100
            tasks:
              greeting:
                prompt: "Say hello"
        """)
        g = load_yaml(path, {})
        assert len(g) == 1
        task = g.tasks["greeting"]
        assert isinstance(task, LLMTask)
        assert task.prompt == "Say hello"
        assert task.model == "claude-sonnet-4-6"
        assert task.max_tokens == 100

    def test_foreach_expansion(self):
        path = _write_yaml("""\
            tasks:
              summarize:
                foreach: $items
                prompt: |
                  Summarize: {{ item }}
        """)
        g = load_yaml(path, {"items": ["a", "b", "c"]})
        assert len(g) == 3
        assert "summarize[0]" in g.task_ids
        assert "summarize[1]" in g.task_ids
        assert "summarize[2]" in g.task_ids

        task = g.tasks["summarize[0]"]
        assert isinstance(task, LLMTask)
        assert "Summarize: a" in task.prompt

    def test_foreach_with_index_deps(self):
        path = _write_yaml("""\
            tasks:
              step1:
                foreach: $items
                prompt: "Do {{ item }}"
              step2:
                foreach: $items
                deps: ["step1[{index}]"]
                prompt: "Follow up on {{ item }}"
        """)
        g = load_yaml(path, {"items": ["x", "y"]})
        assert len(g) == 4
        task = g.tasks["step2[0]"]
        assert task.deps == ["step1[0]"]
        task = g.tasks["step2[1]"]
        assert task.deps == ["step1[1]"]

    def test_foreach_with_star_deps(self):
        path = _write_yaml("""\
            tasks:
              step1:
                foreach: $items
                prompt: "Do {{ item }}"
              report:
                type: python
                deps: ["step1[*]"]
                fn: "builtins:len"
        """)
        g = load_yaml(path, {"items": ["a", "b", "c"]})
        task = g.tasks["report"]
        assert isinstance(task, PyTask)
        assert set(task.deps) == {"step1[0]", "step1[1]", "step1[2]"}

    def test_missing_variable_raises(self):
        path = _write_yaml("""\
            tasks:
              step1:
                foreach: $missing
                prompt: "{{ item }}"
        """)
        with pytest.raises(TemplateError, match="missing"):
            load_yaml(path, {})

    def test_defaults_apply(self):
        path = _write_yaml("""\
            defaults:
              model: claude-haiku-4-5-20251001
              max_tokens: 50
            tasks:
              t1:
                prompt: "Hi"
              t2:
                model: claude-sonnet-4-6
                prompt: "Bye"
        """)
        g = load_yaml(path, {})
        assert g.tasks["t1"].model == "claude-haiku-4-5-20251001"
        assert g.tasks["t1"].max_tokens == 50
        assert g.tasks["t2"].model == "claude-sonnet-4-6"
