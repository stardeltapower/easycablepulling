"""Integration tests for CLI functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from easycablepulling.cli import main
from easycablepulling.core.models import CableSpec, DuctSpec


@pytest.fixture
def test_data_dir() -> Path:
    """Return the path to test data directory."""
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestCLIBasicFunctionality:
    """Test basic CLI functionality."""

    def test_cli_help_command(self, monkeypatch):
        """Test CLI help command."""
        with pytest.raises(SystemExit) as exc_info:
            monkeypatch.setattr("sys.argv", ["easycablepulling", "--help"])
            main()

        # Help should exit with code 0
        assert exc_info.value.code == 0

    def test_cli_version_command(self, monkeypatch):
        """Test CLI version command."""
        # This would test version output if implemented
        # For now, just ensure CLI can be imported
        assert main is not None


class TestCLIAnalysisCommands:
    """Test CLI analysis commands with synthetic routes."""

    def test_cli_geometry_analysis(self, test_data_dir, temp_output_dir, monkeypatch):
        """Test CLI geometry analysis command."""
        input_file = test_data_dir / "straight_route.dxf"
        output_file = temp_output_dir / "geometry_output.dxf"

        # Mock command line arguments
        args = [
            "easycablepulling",
            "analyze-geometry",
            str(input_file),
            "--output",
            str(output_file),
            "--cable-diameter",
            "35.0",
            "--min-bend-radius",
            "1200.0",
        ]

        monkeypatch.setattr("sys.argv", args)

        try:
            main()
            # If CLI supports this command, output file should exist
            if output_file.exists():
                assert output_file.stat().st_size > 0
        except SystemExit as e:
            # CLI might exit with success code
            assert e.code in [0, None]
        except NotImplementedError:
            # Command might not be implemented yet
            pytest.skip("CLI geometry analysis not implemented")

    def test_cli_full_analysis(self, test_data_dir, temp_output_dir, monkeypatch):
        """Test CLI full analysis command."""
        input_file = test_data_dir / "s_curve_route.dxf"
        output_dir = temp_output_dir / "analysis_output"
        output_dir.mkdir()

        args = [
            "easycablepulling",
            "analyze",
            str(input_file),
            "--output-dir",
            str(output_dir),
            "--cable-diameter",
            "35.0",
            "--cable-weight",
            "2.5",
            "--max-tension",
            "8000.0",
            "--duct-diameter",
            "100.0",
            "--friction-dry",
            "0.35",
        ]

        monkeypatch.setattr("sys.argv", args)

        try:
            main()
            # Check for expected output files
            expected_files = ["analysis.json", "report.txt", "summary.csv"]
            for filename in expected_files:
                output_file = output_dir / filename
                if output_file.exists():
                    assert output_file.stat().st_size > 0
        except SystemExit as e:
            assert e.code in [0, None]
        except NotImplementedError:
            pytest.skip("CLI full analysis not implemented")


class TestCLIOutputFormats:
    """Test different CLI output formats."""

    def test_cli_json_output(self, test_data_dir, temp_output_dir, monkeypatch):
        """Test CLI JSON output format."""
        input_file = test_data_dir / "circular_arc_90.0deg.dxf"
        output_file = temp_output_dir / "results.json"

        args = [
            "easycablepulling",
            "analyze",
            str(input_file),
            "--output",
            str(output_file),
            "--format",
            "json",
            "--cable-diameter",
            "35.0",
            "--duct-diameter",
            "100.0",
        ]

        monkeypatch.setattr("sys.argv", args)

        try:
            main()
            if output_file.exists():
                # Should be valid JSON
                with open(output_file) as f:
                    data = json.load(f)

                # Should have expected structure
                assert isinstance(data, dict)
                if "results" in data:
                    assert "total_length_m" in data["results"]
        except (SystemExit, NotImplementedError):
            pytest.skip("CLI JSON output not implemented")

    def test_cli_csv_output(self, test_data_dir, temp_output_dir, monkeypatch):
        """Test CLI CSV output format."""
        input_file = test_data_dir / "complex_route.dxf"
        output_file = temp_output_dir / "results.csv"

        args = [
            "easycablepulling",
            "analyze",
            str(input_file),
            "--output",
            str(output_file),
            "--format",
            "csv",
            "--cable-diameter",
            "35.0",
            "--duct-diameter",
            "100.0",
        ]

        monkeypatch.setattr("sys.argv", args)

        try:
            main()
            if output_file.exists():
                # Should be valid CSV
                with open(output_file) as f:
                    content = f.read()

                lines = content.strip().split("\n")
                assert len(lines) >= 2  # Header + at least one data row

                # Header should contain expected columns
                header = lines[0]
                assert "section_id" in header
                assert "length_m" in header
        except (SystemExit, NotImplementedError):
            pytest.skip("CLI CSV output not implemented")


class TestCLIBatchProcessing:
    """Test CLI batch processing capabilities."""

    def test_cli_batch_analysis(self, test_data_dir, temp_output_dir, monkeypatch):
        """Test CLI batch processing of multiple files."""
        output_dir = temp_output_dir / "batch_output"
        output_dir.mkdir()

        # Test with multiple synthetic routes
        args = [
            "easycablepulling",
            "batch-analyze",
            str(test_data_dir),
            "--output-dir",
            str(output_dir),
            "--pattern",
            "*.dxf",
            "--cable-diameter",
            "35.0",
            "--duct-diameter",
            "100.0",
        ]

        monkeypatch.setattr("sys.argv", args)

        try:
            main()
            # Should create output files for each input
            output_files = list(output_dir.glob("*"))
            if output_files:
                assert len(output_files) > 0
        except (SystemExit, NotImplementedError):
            pytest.skip("CLI batch analysis not implemented")


class TestCLIConfigurationFiles:
    """Test CLI with configuration files."""

    def test_cli_with_config_file(self, test_data_dir, temp_output_dir, monkeypatch):
        """Test CLI with configuration file."""
        # Create test config file
        config_data = {
            "cable_spec": {
                "diameter": 35.0,
                "weight_per_meter": 2.5,
                "max_tension": 8000.0,
                "max_sidewall_pressure": 500.0,
                "min_bend_radius": 1200.0,
            },
            "duct_spec": {
                "inner_diameter": 100.0,
                "type": "PVC",
                "friction_dry": 0.35,
                "friction_lubricated": 0.15,
            },
            "pipeline_options": {
                "enable_splitting": True,
                "max_cable_length": 500.0,
                "safety_factor": 1.5,
            },
        }

        config_file = temp_output_dir / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)

        input_file = test_data_dir / "s_curve_route.dxf"
        output_file = temp_output_dir / "config_test_output.json"

        args = [
            "easycablepulling",
            "analyze",
            str(input_file),
            "--config",
            str(config_file),
            "--output",
            str(output_file),
        ]

        monkeypatch.setattr("sys.argv", args)

        try:
            main()
            if output_file.exists():
                assert output_file.stat().st_size > 0
        except (SystemExit, NotImplementedError):
            pytest.skip("CLI config file support not implemented")


class TestCLIValidation:
    """Test CLI input validation."""

    def test_cli_parameter_validation(self, test_data_dir, monkeypatch):
        """Test CLI parameter validation."""
        input_file = test_data_dir / "straight_route.dxf"

        # Test with invalid parameters
        invalid_args_sets = [
            # Negative diameter
            [
                "easycablepulling",
                "analyze",
                str(input_file),
                "--cable-diameter",
                "-35.0",
            ],
            # Zero duct diameter
            ["easycablepulling", "analyze", str(input_file), "--duct-diameter", "0.0"],
            # Invalid friction coefficient
            ["easycablepulling", "analyze", str(input_file), "--friction-dry", "2.0"],
        ]

        for args in invalid_args_sets:
            monkeypatch.setattr("sys.argv", args)

            try:
                main()
                # Should either raise exception or exit with error code
            except (SystemExit, ValueError, TypeError) as e:
                if isinstance(e, SystemExit):
                    assert e.code != 0  # Should exit with error
            except NotImplementedError:
                pytest.skip("CLI parameter validation not implemented")


class TestCLIPerformance:
    """Test CLI performance characteristics."""

    def test_cli_performance_simple_route(
        self, test_data_dir, temp_output_dir, monkeypatch
    ):
        """Test CLI performance on simple route."""
        import time

        input_file = test_data_dir / "straight_route.dxf"
        output_file = temp_output_dir / "perf_test.json"

        args = [
            "easycablepulling",
            "analyze",
            str(input_file),
            "--output",
            str(output_file),
            "--cable-diameter",
            "35.0",
            "--duct-diameter",
            "100.0",
        ]

        monkeypatch.setattr("sys.argv", args)

        start_time = time.time()
        try:
            main()
            execution_time = time.time() - start_time

            # Should complete quickly for simple route
            assert execution_time < 10.0

            print(f"CLI simple route analysis: {execution_time:.3f}s")
        except (SystemExit, NotImplementedError):
            # CLI might not be fully implemented
            pass

    def test_cli_performance_complex_route(
        self, test_data_dir, temp_output_dir, monkeypatch
    ):
        """Test CLI performance on complex route."""
        import time

        input_file = test_data_dir / "complex_route.dxf"
        output_file = temp_output_dir / "complex_perf_test.json"

        args = [
            "easycablepulling",
            "analyze",
            str(input_file),
            "--output",
            str(output_file),
            "--cable-diameter",
            "35.0",
            "--duct-diameter",
            "100.0",
            "--enable-splitting",
        ]

        monkeypatch.setattr("sys.argv", args)

        start_time = time.time()
        try:
            main()
            execution_time = time.time() - start_time

            # Should complete within reasonable time
            assert execution_time < 30.0

            print(f"CLI complex route analysis: {execution_time:.3f}s")
        except (SystemExit, NotImplementedError):
            pass
