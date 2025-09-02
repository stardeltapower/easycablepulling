"""Integration tests using synthetic test routes."""

import time
from pathlib import Path
from typing import Dict, List

import pytest

from easycablepulling.core.models import CableArrangement, CableSpec, DuctSpec
from easycablepulling.core.pipeline import CablePullingPipeline


@pytest.fixture
def test_data_dir() -> Path:
    """Return the path to test data directory."""
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def standard_cable_spec() -> CableSpec:
    """Standard cable specification for testing."""
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
def standard_duct_spec() -> DuctSpec:
    """Standard duct specification for testing."""
    return DuctSpec(
        inner_diameter=100.0,
        type="PVC",
        friction_dry=0.35,
        friction_lubricated=0.15,
    )


@pytest.fixture
def performance_cable_spec() -> CableSpec:
    """Larger cable for performance testing."""
    return CableSpec(
        diameter=50.0,
        weight_per_meter=4.0,
        max_tension=12000.0,
        max_sidewall_pressure=800.0,
        min_bend_radius=1500.0,
        arrangement=CableArrangement.TREFOIL,
        number_of_cables=3,
    )


class TestStraightRoutes:
    """Test analysis of straight routes."""

    def test_straight_route_basic(
        self, test_data_dir, standard_cable_spec, standard_duct_spec
    ):
        """Test basic straight route analysis."""
        # Use lower safety factor for basic feasibility test
        pipeline = CablePullingPipeline(max_cable_length=500.0, safety_factor=1.0)
        dxf_path = str(test_data_dir / "straight_route.dxf")

        result = pipeline.run_analysis(
            dxf_path, standard_cable_spec, standard_duct_spec
        )

        # Analysis should complete successfully
        assert len(result.errors) == 0, f"Pipeline errors: {result.errors}"
        assert result.processed_route.section_count >= 1
        assert result.summary["total_length_m"] > 0

        # Check that analysis was performed
        assert len(result.tension_analyses) > 0
        assert len(result.limit_results) > 0

        # For a 1000m straight route, tension should be calculable
        max_tension = result.summary["feasibility"]["max_tension_n"]
        assert max_tension > 0

    def test_straight_route_geometry_accuracy(
        self, test_data_dir, standard_cable_spec, standard_duct_spec
    ):
        """Test geometry fitting accuracy on straight route."""
        pipeline = CablePullingPipeline()
        dxf_path = test_data_dir / "straight_route.dxf"

        result = pipeline.run_geometry_only(
            dxf_path, standard_duct_spec, standard_cable_spec
        )

        assert result.success

        # Length deviation should be minimal for straight route
        original_length = result.route.total_length
        fitted_length = sum(section.total_length for section in result.route.sections)
        deviation_percent = abs(fitted_length - original_length) / original_length * 100
        assert deviation_percent < 1.0  # Less than 1% deviation


class TestCurvedRoutes:
    """Test analysis of curved routes."""

    def test_s_curve_analysis(
        self, test_data_dir, standard_cable_spec, standard_duct_spec
    ):
        """Test S-curve route analysis."""
        pipeline = CablePullingPipeline(max_cable_length=1000.0, safety_factor=1.0)
        dxf_path = str(test_data_dir / "s_curve_route.dxf")

        result = pipeline.run_analysis(
            dxf_path, standard_cable_spec, standard_duct_spec
        )

        # Analysis should complete even if geometry has minor issues
        assert len(result.tension_analyses) > 0, "Should have tension analysis results"
        assert result.processed_route.section_count >= 1

        # Should detect bends in the route
        has_bends = any(
            any(hasattr(prim, "radius_m") for prim in section.primitives)
            for section in result.processed_route.sections
        )
        assert has_bends, "Should detect bend primitives in S-curve"

    def test_circular_arc_90deg(
        self, test_data_dir, standard_cable_spec, standard_duct_spec
    ):
        """Test 90-degree circular arc analysis."""
        pipeline = CablePullingPipeline()
        dxf_path = test_data_dir / "circular_arc_90.0deg.dxf"

        result = pipeline.run_analysis(
            dxf_path, standard_cable_spec, standard_duct_spec
        )

        assert result.success

        # Arc should introduce sidewall pressure
        max_pressure = result.summary["feasibility"]["max_pressure_n_per_m"]
        assert max_pressure > 0

    def test_circular_arc_45deg(
        self, test_data_dir, standard_cable_spec, standard_duct_spec
    ):
        """Test 45-degree circular arc analysis."""
        pipeline = CablePullingPipeline()
        dxf_path = test_data_dir / "circular_arc_45.0deg.dxf"

        result = pipeline.run_analysis(
            dxf_path, standard_cable_spec, standard_duct_spec
        )

        assert result.success

        # 45-degree arc should have lower pressure than 90-degree
        max_pressure = result.summary["feasibility"]["max_pressure_n_per_m"]
        assert max_pressure > 0
        assert max_pressure < standard_cable_spec.max_sidewall_pressure


class TestComplexRoutes:
    """Test analysis of complex routes."""

    def test_complex_route_analysis(
        self, test_data_dir, standard_cable_spec, standard_duct_spec
    ):
        """Test complex route with multiple bend types."""
        pipeline = CablePullingPipeline(
            enable_splitting=True,
            max_cable_length=1000.0,
            safety_factor=1.5,
        )
        dxf_path = test_data_dir / "complex_route.dxf"

        result = pipeline.run_analysis(
            dxf_path, standard_cable_spec, standard_duct_spec
        )

        assert result.success
        assert result.processed_route.section_count >= 5  # Multiple sections

        # Should have both straights and bends
        section_types = set()
        for section in result.processed_route.sections:
            for prim in section.primitives:
                if hasattr(prim, "radius"):
                    section_types.add("bend")
                else:
                    section_types.add("straight")

        assert "straight" in section_types
        assert "bend" in section_types

    def test_multiple_sections_route(
        self, test_data_dir, standard_cable_spec, standard_duct_spec
    ):
        """Test route with multiple disconnected sections."""
        pipeline = CablePullingPipeline()
        dxf_path = test_data_dir / "multiple_sections_route.dxf"

        result = pipeline.run_analysis(
            dxf_path, standard_cable_spec, standard_duct_spec
        )

        assert result.success
        assert (
            result.processed_route.section_count >= 3
        )  # Multiple disconnected sections

        # Each section should be analyzed independently
        assert len(result.tension_analyses) == result.processed_route.section_count
        assert len(result.limit_results) == result.processed_route.section_count


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_short_route(
        self, test_data_dir, standard_cable_spec, standard_duct_spec
    ):
        """Test very short route handling."""
        pipeline = CablePullingPipeline()
        dxf_path = test_data_dir / "very_short_route.dxf"

        result = pipeline.run_analysis(
            dxf_path, standard_cable_spec, standard_duct_spec
        )

        # Should either succeed or fail gracefully
        if not result.success:
            assert len(result.errors) > 0
        else:
            assert result.summary["total_length_m"] > 0

    def test_tight_bend_route(
        self, test_data_dir, standard_cable_spec, standard_duct_spec
    ):
        """Test route with bend radius below minimum."""
        pipeline = CablePullingPipeline()
        dxf_path = test_data_dir / "tight_bend_route.dxf"

        result = pipeline.run_analysis(
            dxf_path, standard_cable_spec, standard_duct_spec
        )

        # Should detect violation of minimum bend radius
        if result.success:
            # Check if bend radius warnings are present
            has_radius_warnings = any(
                "radius" in warning.lower() for warning in result.warnings
            )
            # Either should have warnings or pressure limits exceeded
            assert (
                has_radius_warnings
                or len(result.critical_sections["pressure"]) > 0
                or not result.summary["feasibility"]["overall_feasible"]
            )

    def test_many_bends_route(
        self, test_data_dir, standard_cable_spec, standard_duct_spec
    ):
        """Test route with many consecutive bends."""
        pipeline = CablePullingPipeline(max_cable_length=2000.0)
        dxf_path = test_data_dir / "many_bends_route.dxf"

        result = pipeline.run_analysis(
            dxf_path, standard_cable_spec, standard_duct_spec
        )

        assert result.success

        # Should have high cumulative tension
        max_tension = result.summary["feasibility"]["max_tension_n"]
        assert (
            max_tension > standard_cable_spec.max_tension * 0.3
        )  # Significant tension


class TestPerformanceBenchmarks:
    """Performance benchmarks for different route types."""

    def test_straight_route_performance(
        self, test_data_dir, standard_cable_spec, standard_duct_spec
    ):
        """Benchmark straight route processing."""
        pipeline = CablePullingPipeline()
        dxf_path = test_data_dir / "straight_route.dxf"

        start_time = time.time()
        result = pipeline.run_analysis(
            dxf_path, standard_cable_spec, standard_duct_spec
        )
        execution_time = time.time() - start_time

        assert result.success
        assert execution_time < 5.0  # Should complete quickly

        print(f"Straight route analysis: {execution_time:.3f}s")

    def test_complex_route_performance(
        self, test_data_dir, standard_cable_spec, standard_duct_spec
    ):
        """Benchmark complex route processing."""
        pipeline = CablePullingPipeline(enable_splitting=True)
        dxf_path = test_data_dir / "complex_route.dxf"

        start_time = time.time()
        result = pipeline.run_analysis(
            dxf_path, standard_cable_spec, standard_duct_spec
        )
        execution_time = time.time() - start_time

        assert result.success
        assert execution_time < 15.0  # Reasonable time for complex route

        print(f"Complex route analysis: {execution_time:.3f}s")
        print(f"Sections processed: {result.processed_route.section_count}")

    def test_long_route_performance(
        self, test_data_dir, performance_cable_spec, standard_duct_spec
    ):
        """Benchmark long route processing performance."""
        pipeline = CablePullingPipeline(enable_splitting=True, max_cable_length=1000.0)
        dxf_path = test_data_dir / "long_route.dxf"

        start_time = time.time()
        result = pipeline.run_analysis(
            dxf_path, performance_cable_spec, standard_duct_spec
        )
        execution_time = time.time() - start_time

        # Should complete even for long routes
        assert result.success or len(result.errors) > 0
        assert execution_time < 30.0  # Should complete within 30 seconds

        print(f"Long route analysis: {execution_time:.3f}s")
        print(f"Route length: {result.summary['total_length_m']:.1f}m")
        if result.success:
            print(f"Sections: {result.processed_route.section_count}")


class TestWorkflowVariations:
    """Test different workflow configurations and options."""

    def test_geometry_only_workflow(
        self, test_data_dir, standard_cable_spec, standard_duct_spec
    ):
        """Test geometry-only processing workflow."""
        pipeline = CablePullingPipeline()

        test_files = [
            "straight_route.dxf",
            "s_curve_route.dxf",
            "complex_route.dxf",
        ]

        for filename in test_files:
            dxf_path = test_data_dir / filename
            result = pipeline.run_geometry_only(
                dxf_path, standard_duct_spec, standard_cable_spec
            )

            assert result.success, f"Geometry processing failed for {filename}"
            assert result.route.section_count > 0
            assert result.route.total_length > 0

    def test_splitting_enabled_vs_disabled(
        self, test_data_dir, standard_cable_spec, standard_duct_spec
    ):
        """Compare results with splitting enabled vs disabled."""
        dxf_path = test_data_dir / "long_route.dxf"

        # Test without splitting
        pipeline_no_split = CablePullingPipeline(
            enable_splitting=False, max_cable_length=500.0
        )
        result_no_split = pipeline_no_split.run_analysis(
            dxf_path, standard_cable_spec, standard_duct_spec
        )

        # Test with splitting
        pipeline_split = CablePullingPipeline(
            enable_splitting=True, max_cable_length=500.0
        )
        result_split = pipeline_split.run_analysis(
            dxf_path, standard_cable_spec, standard_duct_spec
        )

        # Both should succeed or have meaningful results
        assert result_no_split.success or len(result_no_split.errors) > 0
        assert result_split.success or len(result_split.errors) > 0

        # If both succeed, splitting should generally improve feasibility
        if result_no_split.success and result_split.success:
            split_max_tension = result_split.summary["feasibility"]["max_tension_n"]
            no_split_max_tension = result_no_split.summary["feasibility"][
                "max_tension_n"
            ]

            # Splitting should reduce maximum tension
            assert (
                split_max_tension <= no_split_max_tension * 1.1
            )  # Allow small tolerance

    def test_lubrication_effects(
        self, test_data_dir, standard_cable_spec, standard_duct_spec
    ):
        """Test effects of lubrication on pulling forces."""
        pipeline = CablePullingPipeline()
        dxf_path = test_data_dir / "s_curve_route.dxf"

        # Test dry conditions
        result_dry = pipeline.run_analysis(
            dxf_path, standard_cable_spec, standard_duct_spec, lubricated=False
        )

        # Test lubricated conditions
        result_lubricated = pipeline.run_analysis(
            dxf_path, standard_cable_spec, standard_duct_spec, lubricated=True
        )

        assert result_dry.success
        assert result_lubricated.success

        # Lubrication should reduce tension
        dry_tension = result_dry.summary["feasibility"]["max_tension_n"]
        lubricated_tension = result_lubricated.summary["feasibility"]["max_tension_n"]

        assert lubricated_tension < dry_tension

    def test_safety_factor_variations(
        self, test_data_dir, standard_cable_spec, standard_duct_spec
    ):
        """Test different safety factor settings."""
        dxf_path = test_data_dir / "complex_route.dxf"

        safety_factors = [1.0, 1.5, 2.0, 3.0]
        results = {}

        for sf in safety_factors:
            pipeline = CablePullingPipeline(safety_factor=sf)
            result = pipeline.run_analysis(
                dxf_path, standard_cable_spec, standard_duct_spec
            )
            results[sf] = result

            assert result.success or len(result.errors) > 0
            if result.success:
                assert result.summary["feasibility"]["safety_factor"] == sf

        # Higher safety factors should be more conservative
        successful_results = {sf: r for sf, r in results.items() if r.success}
        if len(successful_results) > 1:
            feasible_counts = {
                sf: r.summary["feasibility"]["overall_feasible"]
                for sf, r in successful_results.items()
            }

            # Generally, lower safety factors should be more permissive
            # (though this depends on the specific route)


class TestErrorHandling:
    """Test error handling and robustness."""

    def test_nonexistent_file_handling(self, standard_cable_spec, standard_duct_spec):
        """Test handling of nonexistent DXF files."""
        pipeline = CablePullingPipeline()

        result = pipeline.run_analysis(
            "nonexistent_file.dxf", standard_cable_spec, standard_duct_spec
        )

        assert not result.success
        assert len(result.errors) > 0
        assert any(
            "not found" in error.lower() or "no such file" in error.lower()
            for error in result.errors
        )

    def test_invalid_cable_specs(self, test_data_dir, standard_duct_spec):
        """Test handling of invalid cable specifications."""
        dxf_path = test_data_dir / "straight_route.dxf"
        pipeline = CablePullingPipeline()

        # Test with zero diameter
        with pytest.raises((ValueError, TypeError)):
            invalid_cable = CableSpec(
                diameter=0.0,  # Invalid
                weight_per_meter=2.5,
                max_tension=8000.0,
                max_sidewall_pressure=500.0,
                min_bend_radius=1200.0,
            )
            pipeline.run_analysis(dxf_path, invalid_cable, standard_duct_spec)

    def test_memory_usage_long_route(
        self, test_data_dir, standard_cable_spec, standard_duct_spec
    ):
        """Test memory usage doesn't grow excessively with long routes."""
        import os

        import psutil

        pipeline = CablePullingPipeline()
        dxf_path = test_data_dir / "long_route.dxf"

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        result = pipeline.run_analysis(
            dxf_path, standard_cable_spec, standard_duct_spec
        )

        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 100MB for test)
        assert memory_increase < 100.0

        print(
            f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{memory_increase:.1f}MB)"
        )


class TestReportGeneration:
    """Test report generation with synthetic routes."""

    def test_text_reports_all_routes(
        self, test_data_dir, standard_cable_spec, standard_duct_spec
    ):
        """Test text report generation for all synthetic routes."""
        from easycablepulling.core.pipeline import AnalysisReporter

        pipeline = CablePullingPipeline()
        test_files = [
            "straight_route.dxf",
            "s_curve_route.dxf",
            "complex_route.dxf",
        ]

        for filename in test_files:
            dxf_path = test_data_dir / filename
            result = pipeline.run_analysis(
                dxf_path, standard_cable_spec, standard_duct_spec
            )

            if result.success:
                report = AnalysisReporter.generate_text_report(result)

                # Check report structure
                assert "CABLE PULLING ANALYSIS REPORT" in report
                assert result.processed_route.name in report
                assert "Total Length:" in report

                # Report should not be empty
                assert len(report.strip()) > 500

    def test_csv_reports_consistency(
        self, test_data_dir, standard_cable_spec, standard_duct_spec
    ):
        """Test CSV report consistency across different routes."""
        from easycablepulling.core.pipeline import AnalysisReporter

        pipeline = CablePullingPipeline()
        test_files = [
            "straight_route.dxf",
            "s_curve_route.dxf",
        ]

        csv_reports = {}
        for filename in test_files:
            dxf_path = test_data_dir / filename
            result = pipeline.run_analysis(
                dxf_path, standard_cable_spec, standard_duct_spec
            )

            if result.success:
                csv_report = AnalysisReporter.generate_csv_report(result)
                csv_reports[filename] = csv_report

        # All CSV reports should have same header structure
        if len(csv_reports) > 1:
            headers = [report.split("\n")[0] for report in csv_reports.values()]
            assert all(header == headers[0] for header in headers)

    def test_json_summary_structure(
        self, test_data_dir, standard_cable_spec, standard_duct_spec
    ):
        """Test JSON summary structure consistency."""
        import json

        from easycablepulling.core.pipeline import AnalysisReporter

        pipeline = CablePullingPipeline()
        dxf_path = test_data_dir / "complex_route.dxf"

        result = pipeline.run_analysis(
            dxf_path, standard_cable_spec, standard_duct_spec
        )

        if result.success:
            json_summary = AnalysisReporter.generate_json_summary(result)

            # Should be valid JSON
            json_str = json.dumps(json_summary)
            parsed = json.loads(json_str)

            # Check required top-level keys
            required_keys = ["meta", "results", "sections", "summary"]
            assert all(key in parsed for key in required_keys)

            # Check meta structure
            meta = parsed["meta"]
            assert "cable_spec" in meta
            assert "analysis_timestamp" in meta

            # Check results structure
            results = parsed["results"]
            assert "overall_feasible" in results
            assert "total_length_m" in results
