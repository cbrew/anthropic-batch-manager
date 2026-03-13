"""Tests for task data models."""

import pytest

from batch_compiler.task import LLMTask, PyTask, TaskResult


class TestLLMTask:
    def test_basic_creation(self):
        t = LLMTask(id="t1", prompt="Hello")
        assert t.id == "t1"
        assert t.prompt == "Hello"
        assert t.deps == []
        assert t.model == "claude-sonnet-4-6"

    def test_with_template(self):
        t = LLMTask(id="t1", prompt_template="Classify: {dep1}")
        assert t.prompt_template == "Classify: {dep1}"
        assert t.prompt is None

    def test_neither_prompt_nor_template_raises(self):
        with pytest.raises(ValueError, match="must provide either"):
            LLMTask(id="t1")

    def test_both_prompt_and_template_raises(self):
        with pytest.raises(ValueError, match="not both"):
            LLMTask(id="t1", prompt="Hi", prompt_template="Hi {dep}")

    def test_with_deps(self):
        t = LLMTask(id="t2", prompt="Go", deps=["t1"])
        assert t.deps == ["t1"]


class TestPyTask:
    def test_basic_creation(self):
        t = PyTask(id="py1", fn=lambda x: x)
        assert t.id == "py1"
        assert t.deps == []

    def test_with_deps(self):
        t = PyTask(id="py1", fn=lambda x: x, deps=["a", "b"])
        assert t.deps == ["a", "b"]


class TestTaskResult:
    def test_succeeded(self):
        r = TaskResult(task_id="t1", status="succeeded", content="Hello world")
        assert r.status == "succeeded"
        assert r.content == "Hello world"
        assert r.error is None

    def test_errored(self):
        r = TaskResult(task_id="t1", status="errored", error="API failure")
        assert r.status == "errored"
        assert r.content is None

    def test_skipped(self):
        r = TaskResult(task_id="t1", status="skipped", error="Dep failed")
        assert r.status == "skipped"
