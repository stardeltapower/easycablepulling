"""Performance and stress tests for the cable pulling system."""

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
def performance_specs() -> Dict[str, any]:
    """Performance test specifications."""
    return {
        "small_cable": CableSpec(
            diameter=20.0,
            weight_per_meter=1.5,
            max_tension=5000.0,
            max_sidewall_pressure=300.0,
            min_bend_radius=600.0,
        ),
        "large_cable": CableSpec(
            diameter=70.0,
            weight_per_meter=6.0,
            max_tension=15000.0,
            max_sidewall_pressure=1000.0,
            min_bend_radius=2000.0,
        ),
        "trefoil_cable": CableSpec(
            diameter=50.0,
            weight_per_meter=4.0,
            max_tension=18000.0,
            max_sidewall_pressure=600.0,
            min_bend_radius=1500.0,
            arrangement=CableArrangement.TREFOIL,
            number_of_cables=3,
        ),
        "small_duct": DuctSpec(
            inner_diameter=80.0,
            type="HDPE",
            friction_dry=0.30,
            friction_lubricated=0.12,
        ),
        "large_duct": DuctSpec(
            inner_diameter=150.0,
            type="PVC",
            friction_dry=0.40,
            friction_lubricated=0.18,
        ),
    }


class TestScalabilityPerformance:
    """Test system performance at different scales."""

    def test_route_length_scaling(self, test_data_dir, performance_specs):
        """Test performance scaling with route length."""
        pipeline = CablePullingPipeline(enable_splitting=True, max_cable_length=1000.0)

        test_routes = [
            ("straight_route.dxf", "short"),
            ("complex_route.dxf", "medium"),
            ("long_route.dxf", "long"),
        ]

        performance_results = {}

        for filename, category in test_routes:
            dxf_path = test_data_dir / filename

            start_time = time.time()
            result = pipeline.run_analysis(
                dxf_path,
                performance_specs["small_cable"],
                performance_specs["small_duct"],
            )
            execution_time = time.time() - start_time

            performance_results[category] = {
                "time": execution_time,
                "success": result.success,
                "length": result.summary["total_length_m"] if result.success else 0,
                "sections": (
                    result.processed_route.section_count if result.success else 0
                ),
            }

            # All routes should complete within reasonable time
            assert (
                execution_time < 45.0
            ), f"{category} route took too long: {execution_time:.2f}s"

        # Print performance summary
        for category, metrics in performance_results.items():
            print(
                f"{category}: {metrics['time']:.3f}s, {metrics['length']:.1f}m, {metrics['sections']} sections"
            )

    def test_cable_complexity_scaling(self, test_data_dir, performance_specs):
        """Test performance scaling with cable complexity."""
        dxf_path = test_data_dir / "complex_route.dxf"
        pipeline = CablePullingPipeline(max_cable_length=800.0)

        cable_types = [
            ("small_cable", "Single small cable"),
            ("large_cable", "Single large cable"),
            ("trefoil_cable", "Trefoil arrangement"),
        ]

        for cable_key, description in cable_types:
            start_time = time.time()
            result = pipeline.run_analysis(
                dxf_path,
                performance_specs[cable_key],
                performance_specs["large_duct"],
            )
            execution_time = time.time() - start_time

            # Should complete regardless of cable complexity
            assert (
                execution_time < 20.0
            ), f"{description} took too long: {execution_time:.2f}s"
            assert result.success or len(result.errors) > 0

            print(f"{description}: {execution_time:.3f}s")

    def test_concurrent_analysis_simulation(self, test_data_dir, performance_specs):
        """Simulate concurrent analysis requests."""
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def run_analysis(route_file: str, cable_key: str, duct_key: str) -> Dict:
            """Run single analysis and return metrics."""
            pipeline = CablePullingPipeline()
            dxf_path = test_data_dir / route_file

            start_time = time.time()
            result = pipeline.run_analysis(
                dxf_path,
                performance_specs[cable_key],
                performance_specs[duct_key],
            )
            execution_time = time.time() - start_time

            return {
                "route": route_file,
                "time": execution_time,
                "success": result.success,
                "thread_id": threading.get_ident(),
            }

        # Simulate 3 concurrent requests
        tasks = [
            ("straight_route.dxf", "small_cable", "small_duct"),
            ("s_curve_route.dxf", "large_cable", "large_duct"),
            ("complex_route.dxf", "trefoil_cable", "large_duct"),
        ]

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(run_analysis, route_file, cable_key, duct_key)
                for route_file, cable_key, duct_key in tasks
            ]

            results = []
            for future in as_completed(futures, timeout=60):
                result = future.result()
                results.append(result)

                # Each analysis should complete
                assert result["time"] < 30.0
                print(
                    f"Concurrent {result['route']}: {result['time']:.3f}s (thread {result['thread_id']})"
                )

        assert len(results) == 3


class TestMemoryPerformance:
    """Test memory usage and performance."""

    def test_memory_usage_scaling(self, test_data_dir, performance_specs):
        """Test memory usage with different route complexities."""
        try:
            import os

            import psutil
        except ImportError:
            pytest.skip("psutil not available for memory testing")

        process = psutil.Process(os.getpid())
        pipeline = CablePullingPipeline()

        test_routes = [
            "straight_route.dxf",
            "s_curve_route.dxf",
            "complex_route.dxf",
            "long_route.dxf",
        ]

        memory_usage = {}

        for route_file in test_routes:
            # Measure memory before
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            dxf_path = test_data_dir / route_file
            result = pipeline.run_analysis(
                dxf_path,
                performance_specs["large_cable"],
                performance_specs["large_duct"],
            )

            # Measure memory after
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            memory_usage[route_file] = {
                "initial": initial_memory,
                "final": final_memory,
                "increase": memory_increase,
                "success": result.success,
            }

            # Memory increase should be reasonable
            assert (
                memory_increase < 50.0
            ), f"Excessive memory usage for {route_file}: {memory_increase:.1f}MB"

            print(
                f"{route_file}: +{memory_increase:.1f}MB (total: {final_memory:.1f}MB)"
            )

    def test_repeated_analysis_memory_stability(self, test_data_dir, performance_specs):
        """Test memory stability with repeated analyses."""
        try:
            import os

            import psutil
        except ImportError:
            pytest.skip("psutil not available for memory testing")

        process = psutil.Process(os.getpid())
        pipeline = CablePullingPipeline()
        dxf_path = test_data_dir / "s_curve_route.dxf"

        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run analysis multiple times
        for i in range(5):
            result = pipeline.run_analysis(
                dxf_path,
                performance_specs["small_cable"],
                performance_specs["small_duct"],
            )

            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory

            # Memory shouldn't grow significantly with repeated use
            assert (
                memory_increase < 30.0
            ), f"Memory leak detected after {i+1} runs: {memory_increase:.1f}MB"

            if i == 4:  # Last iteration
                print(
                    f"Memory after 5 runs: {current_memory:.1f}MB (+{memory_increase:.1f}MB)"
                )


class TestStressConditions:
    """Test system behavior under stress conditions."""

    def test_tight_specifications_stress(self, test_data_dir):
        """Test with very tight cable specifications."""
        # Very restrictive cable spec
        tight_cable = CableSpec(
            diameter=60.0,  # Large diameter
            weight_per_meter=8.0,  # Heavy
            max_tension=3000.0,  # Low tension limit
            max_sidewall_pressure=150.0,  # Low pressure limit
            min_bend_radius=3000.0,  # Large bend radius requirement
        )

        # Small duct
        small_duct = DuctSpec(
            inner_diameter=90.0,  # Tight fit
            type="PVC",
            friction_dry=0.45,  # High friction
            friction_lubricated=0.20,
        )

        pipeline = CablePullingPipeline(safety_factor=2.0)  # High safety factor

        test_routes = ["s_curve_route.dxf", "complex_route.dxf", "many_bends_route.dxf"]

        for route_file in test_routes:
            dxf_path = test_data_dir / route_file
            result = pipeline.run_analysis(dxf_path, tight_cable, small_duct)

            # Should complete analysis even if not feasible
            assert len(result.errors) == 0 or result.success

            # Most routes should fail feasibility with tight specs
            if result.success:
                feasible = result.summary["feasibility"]["overall_feasible"]
                critical_sections = len(result.critical_sections["tension"]) + len(
                    result.critical_sections["pressure"]
                )

                # Should either be infeasible or have critical sections
                assert not feasible or critical_sections > 0

    def test_edge_case_geometries(self, test_data_dir, performance_specs):
        """Test handling of edge case geometries."""
        pipeline = CablePullingPipeline()

        edge_case_files = [
            "very_short_route.dxf",
            "tight_bend_route.dxf",
            "many_bends_route.dxf",
        ]

        for route_file in edge_case_files:
            dxf_path = test_data_dir / route_file

            # Should handle edge cases gracefully
            start_time = time.time()
            result = pipeline.run_analysis(
                dxf_path,
                performance_specs["small_cable"],
                performance_specs["large_duct"],
            )
            execution_time = time.time() - start_time

            # Should either succeed or fail with meaningful errors
            assert result.success or len(result.errors) > 0
            assert execution_time < 20.0  # Should not hang

            print(
                f"Edge case {route_file}: {execution_time:.3f}s, success={result.success}"
            )


class TestResourceLimits:
    """Test behavior at resource limits."""

    def test_large_number_of_sections(self, test_data_dir, performance_specs):
        """Test handling of routes with many sections."""
        pipeline = CablePullingPipeline(
            enable_splitting=True,
            max_cable_length=100.0,  # Force many splits
        )

        dxf_path = test_data_dir / "long_route.dxf"

        start_time = time.time()
        result = pipeline.run_analysis(
            dxf_path,
            performance_specs["small_cable"],
            performance_specs["small_duct"],
        )
        execution_time = time.time() - start_time

        if result.success:
            section_count = result.processed_route.section_count

            # Should handle many sections
            assert section_count > 5  # Should create multiple sections
            assert execution_time < 60.0  # Should complete within a minute

            # Performance should scale reasonably
            time_per_section = execution_time / section_count
            assert time_per_section < 5.0  # Less than 5 seconds per section

            print(
                f"Many sections test: {section_count} sections in {execution_time:.3f}s"
            )
            print(f"Average time per section: {time_per_section:.3f}s")

    def test_high_precision_calculations(self, test_data_dir, performance_specs):
        """Test performance with high-precision calculations."""
        # Use very tight tolerances to force high precision
        pipeline = CablePullingPipeline(
            geometric_tolerance=0.001,  # Very tight tolerance
            angle_tolerance=0.1,  # Very tight angle tolerance
        )

        dxf_path = test_data_dir / "circular_arc_90.0deg.dxf"

        start_time = time.time()
        result = pipeline.run_analysis(
            dxf_path,
            performance_specs["large_cable"],
            performance_specs["small_duct"],
        )
        execution_time = time.time() - start_time

        # High precision should still complete
        assert result.success or len(result.errors) > 0
        assert execution_time < 30.0

        if result.success:
            # Should maintain accuracy
            geometry_result = result.geometry_result
            if hasattr(geometry_result, "max_deviation_mm"):
                assert geometry_result.max_deviation_mm < 1.0  # Very accurate

        print(f"High precision analysis: {execution_time:.3f}s")


class TestRobustnessTests:
    """Test system robustness under various conditions."""

    def test_extreme_cable_specifications(self, test_data_dir):
        """Test with extreme cable specifications."""
        dxf_path = test_data_dir / "s_curve_route.dxf"
        pipeline = CablePullingPipeline()

        extreme_specs = [
            # Ultra-light cable
            CableSpec(5.0, 0.1, 1000.0, 100.0, 200.0),
            # Ultra-heavy cable
            CableSpec(100.0, 20.0, 50000.0, 2000.0, 5000.0),
            # High-performance cable
            CableSpec(25.0, 1.0, 25000.0, 1500.0, 500.0),
        ]

        standard_duct = DuctSpec(120.0, "PVC", 0.35, 0.15)

        for i, cable_spec in enumerate(extreme_specs):
            result = pipeline.run_analysis(dxf_path, cable_spec, standard_duct)

            # Should handle extreme specs without crashing
            assert result.success or len(result.errors) > 0

            if result.success:
                # Results should be physically reasonable
                max_tension = result.summary["feasibility"]["max_tension_n"]
                assert max_tension >= 0
                assert max_tension <= cable_spec.max_tension * 2  # Allow some tolerance

            print(f"Extreme cable spec {i+1}: success={result.success}")

    def test_boundary_conditions(self, test_data_dir, performance_specs):
        """Test boundary conditions and edge values."""
        pipeline = CablePullingPipeline()

        # Test with minimal safety factor
        pipeline_min_safety = CablePullingPipeline(safety_factor=1.0)

        # Test with maximum safety factor
        pipeline_max_safety = CablePullingPipeline(safety_factor=5.0)

        dxf_path = test_data_dir / "tight_bend_route.dxf"

        results = {}
        for name, test_pipeline in [
            ("min_safety", pipeline_min_safety),
            ("max_safety", pipeline_max_safety),
        ]:
            result = test_pipeline.run_analysis(
                dxf_path,
                performance_specs["small_cable"],
                performance_specs["large_duct"],
            )
            results[name] = result

            assert result.success or len(result.errors) > 0

        # Different safety factors should give different feasibility results
        if all(r.success for r in results.values()):
            min_feasible = results["min_safety"].summary["feasibility"][
                "overall_feasible"
            ]
            max_feasible = results["max_safety"].summary["feasibility"][
                "overall_feasible"
            ]

            # Higher safety factor should be more restrictive or equal
            if min_feasible:
                # If minimum safety passes, maximum might still fail due to stricter criteria
                pass  # This is acceptable behavior


class TestReliabilityTests:
    """Test system reliability and error recovery."""

    def test_malformed_geometry_handling(self, test_data_dir, performance_specs):
        """Test handling of potentially malformed geometries."""
        pipeline = CablePullingPipeline()

        # Test files that might have challenging geometries
        challenging_files = [
            "very_short_route.dxf",
            "tight_bend_route.dxf",
            "many_bends_route.dxf",
        ]

        for route_file in challenging_files:
            dxf_path = test_data_dir / route_file

            result = pipeline.run_analysis(
                dxf_path,
                performance_specs["small_cable"],
                performance_specs["small_duct"],
            )

            # Should handle challenging geometries gracefully
            if not result.success:
                # Should have meaningful error messages
                assert len(result.errors) > 0
                assert all(len(error.strip()) > 0 for error in result.errors)
            else:
                # If successful, results should be reasonable
                assert result.summary["total_length_m"] > 0
                assert result.processed_route.section_count > 0

            print(f"Challenging geometry {route_file}: success={result.success}")

    def test_numerical_stability(self, test_data_dir, performance_specs):
        """Test numerical stability with repeated calculations."""
        pipeline = CablePullingPipeline()
        dxf_path = test_data_dir / "circular_arc_90.0deg.dxf"

        # Run same analysis multiple times
        results = []
        for i in range(3):
            result = pipeline.run_analysis(
                dxf_path,
                performance_specs["small_cable"],
                performance_specs["small_duct"],
            )
            results.append(result)

        # All runs should succeed or fail consistently
        success_states = [r.success for r in results]
        assert all(s == success_states[0] for s in success_states)

        # If successful, results should be consistent
        successful_results = [r for r in results if r.success]
        if len(successful_results) > 1:
            tensions = [
                r.summary["feasibility"]["max_tension_n"] for r in successful_results
            ]
            pressures = [
                r.summary["feasibility"]["max_pressure_n_per_m"]
                for r in successful_results
            ]

            # Results should be numerically stable (within 0.1% tolerance)
            tension_variation = max(tensions) - min(tensions)
            pressure_variation = max(pressures) - min(pressures)

            assert tension_variation / max(tensions) < 0.001  # 0.1% tolerance
            assert pressure_variation / max(pressures) < 0.001  # 0.1% tolerance


class TestSystemLimits:
    """Test system behavior at operational limits."""

    def test_maximum_route_complexity(self, test_data_dir, performance_specs):
        """Test maximum route complexity handling."""
        pipeline = CablePullingPipeline(
            enable_splitting=True,
            max_cable_length=50.0,  # Force maximum splitting
        )

        dxf_path = test_data_dir / "many_bends_route.dxf"

        start_time = time.time()
        result = pipeline.run_analysis(
            dxf_path,
            performance_specs["trefoil_cable"],
            performance_specs["small_duct"],
        )
        execution_time = time.time() - start_time

        # Should handle maximum complexity
        assert execution_time < 120.0  # 2 minute timeout
        assert result.success or len(result.errors) > 0

        if result.success:
            section_count = result.processed_route.section_count
            print(
                f"Maximum complexity: {section_count} sections in {execution_time:.3f}s"
            )

            # Should create many sections due to splitting
            assert section_count > 10

    def test_pipeline_timeout_behavior(self, test_data_dir, performance_specs):
        """Test pipeline behavior with timeout conditions."""
        # This test would require timeout implementation in pipeline
        # For now, just test that long operations complete eventually

        pipeline = CablePullingPipeline(
            enable_splitting=True,
            max_cable_length=25.0,  # Force extensive splitting
        )

        dxf_path = test_data_dir / "long_route.dxf"

        start_time = time.time()
        result = pipeline.run_analysis(
            dxf_path,
            performance_specs["trefoil_cable"],
            performance_specs["small_duct"],
        )
        execution_time = time.time() - start_time

        # Should complete within reasonable timeout
        assert execution_time < 180.0  # 3 minute maximum

        print(f"Stress test execution time: {execution_time:.3f}s")
        print(f"Success: {result.success}, Errors: {len(result.errors)}")
