"""Tests for the workflow command."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from datasety.workflow import _args_to_argv, _find_workflow_file, _load_workflow

# ── Unit tests ──


class TestArgsToArgv:
    def test_basic(self):
        argv = _args_to_argv("resize", {"input": "./raw", "resolution": "768x1024"})
        assert argv == ["resize", "--input", "./raw", "--resolution", "768x1024"]

    def test_bool_true(self):
        argv = _args_to_argv("caption", {"llm-api": True, "input": "./imgs"})
        assert "--llm-api" in argv
        assert "True" not in argv

    def test_bool_false(self):
        argv = _args_to_argv("caption", {"llm-api": False, "input": "./imgs"})
        assert "--llm-api" not in argv

    def test_list_values(self):
        argv = _args_to_argv("shuffle", {
            "input": "./in",
            "output": "./out",
            "group": ["A|B", "C|D"],
        })
        assert argv.count("--group") == 2
        assert "A|B" in argv
        assert "C|D" in argv

    def test_numeric_values(self):
        argv = _args_to_argv("resize", {"resolution": "768x1024", "output": "./out"})
        assert "--resolution" in argv
        assert "768x1024" in argv


class TestFindWorkflowFile:
    def test_explicit_path(self, tmp_path):
        f = tmp_path / "my-workflow.yaml"
        f.write_text("steps: []")
        result = _find_workflow_file(str(f))
        assert result == f

    def test_missing_explicit_path(self, tmp_path):
        with pytest.raises(SystemExit):
            _find_workflow_file(str(tmp_path / "nonexistent.yaml"))

    def test_auto_detect_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "datasety.yaml"
        f.write_text("steps: []")
        result = _find_workflow_file()
        assert result == Path("datasety.yaml")

    def test_auto_detect_yml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "datasety.yml"
        f.write_text("steps: []")
        result = _find_workflow_file()
        assert result == Path("datasety.yml")

    def test_auto_detect_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "datasety.json"
        f.write_text('{"steps": []}')
        result = _find_workflow_file()
        assert result == Path("datasety.json")

    def test_no_file_exits(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit):
            _find_workflow_file()


class TestLoadWorkflow:
    def test_load_json(self, tmp_path):
        f = tmp_path / "workflow.json"
        data = {"steps": [{"command": "resize", "args": {"resolution": "512x512"}}]}
        f.write_text(json.dumps(data))
        result = _load_workflow(f)
        assert result == data

    def test_load_yaml(self, tmp_path):
        pytest.importorskip("yaml")
        f = tmp_path / "workflow.yaml"
        f.write_text("steps:\n  - command: resize\n    args:\n      resolution: 512x512\n")
        result = _load_workflow(f)
        assert len(result["steps"]) == 1
        assert result["steps"][0]["command"] == "resize"


# ── CLI tests ──


def run_workflow(*args):
    return subprocess.run(
        [sys.executable, "-m", "datasety", "workflow", *args],
        capture_output=True, text=True,
    )


class TestWorkflowCLI:
    def test_help(self):
        result = run_workflow("--help")
        assert result.returncode == 0
        assert "--file" in result.stdout
        assert "--dry-run" in result.stdout

    def test_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = run_workflow()
        assert result.returncode != 0

    def test_dry_run_valid(self, tmp_path):
        """Dry run with a valid workflow should pass validation."""
        # Create input directories
        in_dir = tmp_path / "raw"
        in_dir.mkdir()
        Image.new("RGB", (100, 100)).save(in_dir / "test.jpg")

        out_dir = tmp_path / "resized"

        workflow_data = {
            "steps": [
                {
                    "command": "resize",
                    "args": {
                        "input": str(in_dir),
                        "output": str(out_dir),
                        "resolution": "64x64",
                    },
                },
            ],
        }

        wf_file = tmp_path / "datasety.json"
        wf_file.write_text(json.dumps(workflow_data))

        result = run_workflow("--file", str(wf_file), "--dry-run")
        assert result.returncode == 0
        assert "validated successfully" in result.stdout

    def test_dry_run_invalid_input(self, tmp_path):
        """Dry run should fail when input dir doesn't exist."""
        workflow_data = {
            "steps": [
                {
                    "command": "resize",
                    "args": {
                        "input": str(tmp_path / "nonexistent"),
                        "output": str(tmp_path / "out"),
                        "resolution": "64x64",
                    },
                },
            ],
        }

        wf_file = tmp_path / "datasety.json"
        wf_file.write_text(json.dumps(workflow_data))

        result = run_workflow("--file", str(wf_file), "--dry-run")
        assert result.returncode != 0
        assert "does not exist" in result.stdout

    def test_execute_workflow(self, tmp_path):
        """Execute a real resize workflow."""
        in_dir = tmp_path / "raw"
        in_dir.mkdir()
        Image.new("RGB", (200, 200)).save(in_dir / "test.jpg")

        out_dir = tmp_path / "resized"

        workflow_data = {
            "steps": [
                {
                    "command": "resize",
                    "args": {
                        "input": str(in_dir),
                        "output": str(out_dir),
                        "resolution": "64x64",
                    },
                },
            ],
        }

        wf_file = tmp_path / "datasety.json"
        wf_file.write_text(json.dumps(workflow_data))

        result = run_workflow("--file", str(wf_file))
        assert result.returncode == 0
        assert (out_dir / "test.jpg").exists()
        img = Image.open(out_dir / "test.jpg")
        assert img.size == (64, 64)

    def test_empty_steps(self, tmp_path):
        """Workflow with no steps should error."""
        wf_file = tmp_path / "datasety.json"
        wf_file.write_text(json.dumps({"steps": []}))

        result = run_workflow("--file", str(wf_file))
        assert result.returncode != 0
        assert "no steps" in result.stdout
