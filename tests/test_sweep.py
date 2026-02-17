"""Tests for the sweep command."""

import subprocess
import sys

import pytest

yaml = pytest.importorskip("yaml")


def run_sweep(*args):
    """Run datasety sweep and return the result."""
    return subprocess.run(
        [sys.executable, "-m", "datasety", "sweep", *args],
        capture_output=True,
        text=True,
    )


class TestSweepCLI:
    """Test sweep CLI argument parsing."""

    def test_help(self):
        result = run_sweep("--help")
        assert result.returncode == 0
        assert "--steps" in result.stdout
        assert "--cfg-scale" in result.stdout
        assert "--output-file" in result.stdout
        assert "--run" in result.stdout

    def test_missing_prompt(self, tmp_path):
        result = run_sweep(
            "-i",
            str(tmp_path),
            "-o",
            str(tmp_path / "out"),
            "--steps",
            "4,8",
        )
        assert result.returncode != 0


class TestSweepGeneration:
    """Test sweep YAML generation."""

    def test_steps_sweep(self, tmp_path):
        """Sweep over steps produces correct number of combinations."""
        output_file = tmp_path / "sweep.yaml"
        result = run_sweep(
            "-i",
            str(tmp_path),
            "-o",
            str(tmp_path / "out"),
            "-p",
            "add a hat",
            "--steps",
            "4,8,16",
            "--output-file",
            str(output_file),
        )
        assert result.returncode == 0
        assert output_file.exists()

        data = yaml.safe_load(output_file.read_text())
        assert len(data["steps"]) == 3
        steps_values = [s["args"]["steps"] for s in data["steps"]]
        assert steps_values == [4, 8, 16]

    def test_cartesian_product(self, tmp_path):
        """Steps x cfg-scale produces Cartesian product."""
        output_file = tmp_path / "sweep.yaml"
        result = run_sweep(
            "-i",
            str(tmp_path),
            "-o",
            str(tmp_path / "out"),
            "-p",
            "test prompt",
            "--steps",
            "4,8",
            "--cfg-scale",
            "1.0,2.5,5.0",
            "--output-file",
            str(output_file),
        )
        assert result.returncode == 0

        data = yaml.safe_load(output_file.read_text())
        assert len(data["steps"]) == 6  # 2 * 3

    def test_output_directory_naming(self, tmp_path):
        """Each step should have a descriptive output subdirectory."""
        output_file = tmp_path / "sweep.yaml"
        run_sweep(
            "-i",
            str(tmp_path),
            "-o",
            str(tmp_path / "out"),
            "-p",
            "test",
            "--steps",
            "4,8",
            "--cfg-scale",
            "1.0,2.5",
            "--output-file",
            str(output_file),
        )
        data = yaml.safe_load(output_file.read_text())

        outputs = [s["args"]["output"] for s in data["steps"]]
        assert "steps4_cfg1.0" in outputs[0]
        assert "steps4_cfg2.5" in outputs[1]
        assert "steps8_cfg1.0" in outputs[2]
        assert "steps8_cfg2.5" in outputs[3]

    def test_passthrough_params(self, tmp_path):
        """Seed and output-format are passed through."""
        output_file = tmp_path / "sweep.yaml"
        run_sweep(
            "-i",
            str(tmp_path),
            "-o",
            str(tmp_path / "out"),
            "-p",
            "test",
            "--steps",
            "4",
            "--seed",
            "42",
            "--output-format",
            "jpg",
            "--output-file",
            str(output_file),
        )
        data = yaml.safe_load(output_file.read_text())

        step = data["steps"][0]
        assert step["args"]["seed"] == 42
        assert step["args"]["output-format"] == "jpg"

    def test_single_value_no_sweep(self, tmp_path):
        """A single value produces exactly one combination."""
        output_file = tmp_path / "sweep.yaml"
        run_sweep(
            "-i",
            str(tmp_path),
            "-o",
            str(tmp_path / "out"),
            "-p",
            "test",
            "--steps",
            "4",
            "--output-file",
            str(output_file),
        )
        data = yaml.safe_load(output_file.read_text())
        assert len(data["steps"]) == 1

    def test_all_steps_are_synthetic(self, tmp_path):
        """All generated steps should use the synthetic command."""
        output_file = tmp_path / "sweep.yaml"
        run_sweep(
            "-i",
            str(tmp_path),
            "-o",
            str(tmp_path / "out"),
            "-p",
            "test",
            "--steps",
            "4,8",
            "--output-file",
            str(output_file),
        )
        data = yaml.safe_load(output_file.read_text())
        for step in data["steps"]:
            assert step["command"] == "synthetic"

    def test_prompt_in_all_steps(self, tmp_path):
        """Prompt should appear in every step."""
        output_file = tmp_path / "sweep.yaml"
        run_sweep(
            "-i",
            str(tmp_path),
            "-o",
            str(tmp_path / "out"),
            "-p",
            "add winter hat",
            "--steps",
            "4,8",
            "--output-file",
            str(output_file),
        )
        data = yaml.safe_load(output_file.read_text())
        for step in data["steps"]:
            assert step["args"]["prompt"] == "add winter hat"

    def test_yaml_header_comments(self, tmp_path):
        """YAML file should contain header comments."""
        output_file = tmp_path / "sweep.yaml"
        run_sweep(
            "-i",
            str(tmp_path),
            "-o",
            str(tmp_path / "out"),
            "-p",
            "test",
            "--steps",
            "4,8",
            "--output-file",
            str(output_file),
        )
        content = output_file.read_text()
        assert "Generated by: datasety sweep" in content
        assert "Total combinations: 2" in content

    def test_no_sweep_params_errors(self, tmp_path):
        """No sweep parameters should produce an error."""
        output_file = tmp_path / "sweep.yaml"
        result = run_sweep(
            "-i",
            str(tmp_path),
            "-o",
            str(tmp_path / "out"),
            "-p",
            "test",
            "--output-file",
            str(output_file),
        )
        assert result.returncode != 0

    def test_strength_sweep(self, tmp_path):
        """Sweep over strength values."""
        output_file = tmp_path / "sweep.yaml"
        run_sweep(
            "-i",
            str(tmp_path),
            "-o",
            str(tmp_path / "out"),
            "-p",
            "test",
            "--strength",
            "0.5,0.7,0.9",
            "--output-file",
            str(output_file),
        )
        data = yaml.safe_load(output_file.read_text())
        assert len(data["steps"]) == 3
        strength_values = [s["args"]["strength"] for s in data["steps"]]
        assert strength_values == [0.5, 0.7, 0.9]
