"""Integration tests for complete analysis pipeline."""

import json
from pathlib import Path

import pytest

from easycablepulling.core.models import CableArrangement, CableSpec, DuctSpec
from easycablepulling.core.pipeline import AnalysisReporter, CablePullingPipeline


@pytest.fixture
def sample_cable_spec():
    """Sample cable specification for testing."""
    return CableSpec(
        diameter=35.0,
        weight_per_meter=2.5,
        max_tension=8000.0,
        max_sidewall_pressure=500.0,
        min_bend_radius=1200.0,
        arrangement=CableArrangement.SINGLE,
        number_of_cables=1,
    )


@pytest.fixture
def sample_duct_spec():
    """Sample duct specification for testing."""
    return DuctSpec(
        inner_diameter=100.0,
        type="PVC",
        friction_dry=0.35,
        friction_lubricated=0.15,
    )


@pytest.fixture
def input_dxf_path():
    """Path to test DXF file."""
    return "examples/input.dxf"


class TestCablePullingPipeline:
    """Test complete cable pulling analysis pipeline."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization with custom parameters."""
        pipeline = CablePullingPipeline(
            enable_splitting=True,
            max_cable_length=400.0,
            safety_factor=2.0,
        )

        assert pipeline.enable_splitting is True
        assert pipeline.max_cable_length == 400.0
        assert pipeline.safety_factor == 2.0

    def test_geometry_only_pipeline(
        self, input_dxf_path, sample_duct_spec, sample_cable_spec
    ):
        """Test geometry-only processing."""
        pipeline = CablePullingPipeline(max_cable_length=500.0)

        result = pipeline.run_geometry_only(
            input_dxf_path,
            duct_spec=sample_duct_spec,
            cable_spec=sample_cable_spec,
        )

        assert result.success
        assert result.route.section_count > 0
        assert result.route.total_length > 0

        # Should have fitted primitives
        total_primitives = sum(
            len(section.primitives) for section in result.route.sections
        )
        assert total_primitives > 0

    def test_complete_analysis_pipeline(
        self, input_dxf_path, sample_cable_spec, sample_duct_spec
    ):
        """Test complete analysis pipeline."""
        pipeline = CablePullingPipeline(
            enable_splitting=True,
            max_cable_length=500.0,
            safety_factor=1.5,
        )

        result = pipeline.run_analysis(
            input_dxf_path,
            sample_cable_spec,
            sample_duct_spec,
            lubricated=False,
        )

        # Check basic structure
        assert hasattr(result, "success")
        assert hasattr(result, "summary")
        assert hasattr(result, "feasibility_result")
        assert hasattr(result, "tension_analyses")
        assert hasattr(result, "limit_results")

        # Check that we have results for all sections
        assert len(result.tension_analyses) == result.processed_route.section_count
        assert len(result.limit_results) == result.processed_route.section_count

        # Check summary structure
        assert "route_name" in result.summary
        assert "total_length_m" in result.summary
        assert "feasibility" in result.summary

    def test_pipeline_with_trefoil_cables(self, input_dxf_path, sample_duct_spec):
        """Test pipeline with trefoil cable arrangement."""
        trefoil_cable = CableSpec(
            diameter=35.0,
            weight_per_meter=2.5,
            max_tension=12000.0,  # Higher limit for trefoil
            max_sidewall_pressure=400.0,
            min_bend_radius=1200.0,
            arrangement=CableArrangement.TREFOIL,
            number_of_cables=3,
        )

        pipeline = CablePullingPipeline(max_cable_length=500.0)

        result = pipeline.run_analysis(
            input_dxf_path,
            trefoil_cable,
            sample_duct_spec,
            lubricated=True,  # Test with lubrication
        )

        # Should handle trefoil arrangement
        assert result.cable_spec.arrangement == CableArrangement.TREFOIL
        assert result.cable_spec.number_of_cables == 3

        # Should have different friction calculations than single cable
        assert result.summary["feasibility"]["max_tension_n"] != 0

    def test_pipeline_error_handling(self):
        """Test pipeline error handling with invalid input."""
        pipeline = CablePullingPipeline()

        # Test with non-existent file
        result = pipeline.run_analysis(
            "nonexistent.dxf",
            CableSpec(35.0, 2.5, 8000.0, 500.0, 1200.0),
            DuctSpec(100.0, "PVC", 0.35, 0.15),
        )

        assert not result.success
        assert len(result.errors) > 0
        assert "Pipeline failed" in result.errors[0]


class TestAnalysisReporter:
    """Test analysis reporting functionality."""

    def test_text_report_generation(
        self, input_dxf_path, sample_cable_spec, sample_duct_spec
    ):
        """Test text report generation."""
        pipeline = CablePullingPipeline(max_cable_length=500.0)
        result = pipeline.run_analysis(
            input_dxf_path, sample_cable_spec, sample_duct_spec
        )

        report = AnalysisReporter.generate_text_report(result)

        # Check report contains key sections
        assert "CABLE PULLING ANALYSIS REPORT" in report
        assert "OVERALL STATUS:" in report
        assert "ROUTE SUMMARY:" in report
        assert "FEASIBILITY ANALYSIS:" in report
        assert "SECTION DETAILS:" in report

        # Check for actual data
        assert "Total Length:" in report
        assert "Max tension:" in report
        assert result.processed_route.name in report

    def test_csv_report_generation(
        self, input_dxf_path, sample_cable_spec, sample_duct_spec
    ):
        """Test CSV report generation."""
        pipeline = CablePullingPipeline(max_cable_length=500.0)
        result = pipeline.run_analysis(
            input_dxf_path, sample_cable_spec, sample_duct_spec
        )

        report = AnalysisReporter.generate_csv_report(result)

        # Check CSV structure
        lines = report.split("\n")
        assert len(lines) > 1  # Header + at least one data row

        # Check header
        header = lines[0]
        expected_columns = [
            "section_id",
            "length_m",
            "max_tension_n",
            "max_pressure_n_per_m",
            "recommended_direction",
            "passes_all_limits",
            "limiting_factors",
        ]
        for column in expected_columns:
            assert column in header

        # Check data rows
        data_lines = lines[1:]
        assert len(data_lines) == result.processed_route.section_count

        # Check first data row has correct number of columns
        first_row = data_lines[0].split(",")
        assert len(first_row) == len(expected_columns)

    def test_json_summary_generation(
        self, input_dxf_path, sample_cable_spec, sample_duct_spec
    ):
        """Test JSON summary generation."""
        pipeline = CablePullingPipeline(max_cable_length=500.0)
        result = pipeline.run_analysis(
            input_dxf_path, sample_cable_spec, sample_duct_spec
        )

        json_summary = AnalysisReporter.generate_json_summary(result)

        # Check JSON structure
        assert "meta" in json_summary
        assert "results" in json_summary
        assert "sections" in json_summary
        assert "summary" in json_summary

        # Check meta information
        meta = json_summary["meta"]
        assert "cable_spec" in meta
        assert "duct_spec" in meta
        assert meta["cable_spec"]["diameter_mm"] == sample_cable_spec.diameter

        # Check results
        results = json_summary["results"]
        assert "overall_feasible" in results
        assert "total_length_m" in results
        assert "max_tension_n" in results

        # Check sections array
        sections = json_summary["sections"]
        assert len(sections) == result.processed_route.section_count
        assert all("id" in section for section in sections)
        assert all("length_m" in section for section in sections)


class TestPipelinePerformance:
    """Test pipeline performance and resource usage."""

    def test_pipeline_performance(
        self, input_dxf_path, sample_cable_spec, sample_duct_spec
    ):
        """Test pipeline completes within reasonable time."""
        import time

        pipeline = CablePullingPipeline(max_cable_length=500.0)

        start_time = time.time()
        result = pipeline.run_analysis(
            input_dxf_path, sample_cable_spec, sample_duct_spec
        )
        end_time = time.time()

        execution_time = end_time - start_time

        # Should complete in reasonable time (< 30 seconds for test file)
        assert execution_time < 30.0
        assert (
            result.success or len(result.errors) > 0
        )  # Should either succeed or have meaningful errors

        # Log performance for reference
        print(f"Pipeline execution time: {execution_time:.2f} seconds")
        print(f"Sections processed: {result.processed_route.section_count}")
        print(f"Total length: {result.summary['total_length_m']:.1f}m")


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_single_cable_workflow(self, input_dxf_path):
        """Test complete workflow for single cable."""
        # Create specifications
        cable_spec = CableSpec(
            diameter=25.0,
            weight_per_meter=1.8,
            max_tension=6000.0,
            max_sidewall_pressure=400.0,
            min_bend_radius=800.0,
            arrangement=CableArrangement.SINGLE,
            number_of_cables=1,
        )

        duct_spec = DuctSpec(
            inner_diameter=80.0,
            type="HDPE",
            friction_dry=0.35,
            friction_lubricated=0.15,
        )

        # Run complete analysis
        pipeline = CablePullingPipeline(
            enable_splitting=True,
            max_cable_length=600.0,
            safety_factor=1.2,
        )

        result = pipeline.run_analysis(input_dxf_path, cable_spec, duct_spec, False)

        # Verify results
        assert result.input_file == input_dxf_path
        assert result.cable_spec == cable_spec
        assert len(result.duct_specs) > 0
        assert result.summary["feasibility"]["safety_factor"] == 1.2

    def test_trefoil_workflow_with_limits(self, input_dxf_path):
        """Test workflow with trefoil cables that may hit limits."""
        # Create trefoil specification with tighter limits
        cable_spec = CableSpec(
            diameter=50.0,  # Larger cable
            weight_per_meter=4.0,  # Heavier
            max_tension=5000.0,  # Lower tension limit
            max_sidewall_pressure=200.0,  # Lower pressure limit
            min_bend_radius=2000.0,  # Larger bend radius
            arrangement=CableArrangement.TREFOIL,
            number_of_cables=3,
        )

        duct_spec = DuctSpec(
            inner_diameter=120.0,
            type="PVC",
            friction_dry=0.35,
            friction_lubricated=0.15,
        )

        pipeline = CablePullingPipeline(max_cable_length=400.0)
        result = pipeline.run_analysis(input_dxf_path, cable_spec, duct_spec, False)

        # This configuration should likely fail some limits
        if not result.success:
            assert (
                len(result.critical_sections["tension"]) > 0
                or len(result.critical_sections["pressure"]) > 0
            )

        # Should still complete analysis even if not feasible
        assert len(result.tension_analyses) > 0
        assert len(result.limit_results) > 0
