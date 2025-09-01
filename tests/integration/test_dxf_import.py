"""Integration tests for DXF import functionality."""

from pathlib import Path

import pytest

from easycablepulling.io.dxf_reader import DXFReader, load_route_from_dxf


class TestDXFImport:
    """Test DXF import functionality with real files."""

    def test_load_input_dxf(self, input_dxf_path) -> None:
        """Test loading the default input.dxf file."""
        reader = DXFReader(input_dxf_path)
        reader.load()

        # Check basic properties
        assert reader.doc is not None

        # Get file summary
        summary = reader.get_route_summary()
        assert summary["file_path"] == str(input_dxf_path)
        assert "layers" in summary
        assert "total_polylines" in summary
        assert summary["total_polylines"] > 0

        print(f"DXF Summary: {summary}")

    def test_extract_polylines_from_input(self, input_dxf_path) -> None:
        """Test extracting polylines from input.dxf."""
        reader = DXFReader(input_dxf_path)
        reader.load()

        polylines = reader.extract_polylines()
        assert len(polylines) > 0

        # Check structure
        for layer_name, points in polylines:
            assert isinstance(layer_name, str)
            assert isinstance(points, list)
            assert len(points) >= 2

            # Check point format
            for x, y in points:
                assert isinstance(x, (int, float))
                assert isinstance(y, (int, float))

        print(f"Found {len(polylines)} polylines")
        for i, (layer, points) in enumerate(polylines):
            print(f"  Polyline {i+1}: Layer '{layer}', {len(points)} points")

    def test_create_route_from_input_dxf(self, input_dxf_path) -> None:
        """Test creating a Route object from input.dxf."""
        route = load_route_from_dxf(input_dxf_path, "Test Route")

        assert route.name == "Test Route"
        assert route.section_count > 0
        # Note: total_length will be 0 until primitives are fitted in Phase 3
        # For now, check that sections have original polyline data
        total_original_length = sum(
            section.original_length for section in route.sections
        )
        assert total_original_length > 0

        # Check metadata
        assert "source_file" in route.metadata
        assert "source_layers" in route.metadata
        assert "polyline_count" in route.metadata

        print(f"Route: {route.name}")
        print(f"Sections: {route.section_count}")
        total_original = sum(section.original_length for section in route.sections)
        print(f"Total original length: {total_original:.2f}m")
        print(f"Metadata: {route.metadata}")

        # Validate each section
        for section in route.sections:
            assert len(section.original_polyline) >= 2
            assert section.original_length > 0
            print(
                f"  Section {section.id}: {len(section.original_polyline)} points, "
                f"{section.original_length:.2f}m"
            )

    def test_layer_filtering(self, input_dxf_path) -> None:
        """Test filtering by specific layer."""
        reader = DXFReader(input_dxf_path)
        reader.load()

        # Get all layers
        layers = reader.get_layers()
        assert len(layers) > 0

        # Test filtering by first layer
        first_layer = layers[0]
        filtered_polylines = reader.extract_polylines(layer_name=first_layer)

        # All polylines should be from the specified layer
        for layer_name, points in filtered_polylines:
            assert layer_name == first_layer

        print(f"Filtered by layer '{first_layer}': {len(filtered_polylines)} polylines")

    def test_angle_calculation(self) -> None:
        """Test angle change calculation."""
        from easycablepulling.io.polyline_parser import PolylineParser

        parser = PolylineParser()

        # Test 90-degree turn
        p1 = (0.0, 0.0)
        p2 = (10.0, 0.0)
        p3 = (10.0, 10.0)

        angle_change = parser._calculate_angle_change(p1, p2, p3)
        assert abs(angle_change - 90.0) < 0.1

        # Test straight line (no angle change)
        p1 = (0.0, 0.0)
        p2 = (10.0, 0.0)
        p3 = (20.0, 0.0)

        angle_change = parser._calculate_angle_change(p1, p2, p3)
        assert abs(angle_change) < 0.1

        print(f"90° turn: {angle_change:.1f}°")

    def test_section_naming(self) -> None:
        """Test section ID generation."""
        from easycablepulling.io.polyline_parser import PolylineParser

        parser = PolylineParser()

        # Test normal alphabetical naming
        assert parser._generate_section_id(0, 3) == "AB"
        assert parser._generate_section_id(1, 3) == "BC"
        assert parser._generate_section_id(2, 3) == "CD"

        # Test fallback naming for many sections
        assert parser._generate_section_id(0, 30) == "SECT_01_02"
        assert parser._generate_section_id(25, 30) == "SECT_26_27"

    def test_round_trip_import_export(self, input_dxf_path, tmp_path) -> None:
        """Test importing DXF and exporting it back."""
        from easycablepulling.io.dxf_writer import export_route_to_dxf

        # Import original
        original_route = load_route_from_dxf(input_dxf_path, "Original Route")

        # Export to temporary file
        output_path = tmp_path / "exported_route.dxf"
        export_route_to_dxf(
            route=original_route,
            file_path=output_path,
            include_annotations=True,
            include_joint_markers=True,
        )

        assert output_path.exists()

        # Import the exported file
        imported_route = load_route_from_dxf(output_path, "Imported Route")

        # Compare basic properties
        assert imported_route.section_count >= original_route.section_count

        # The imported route should have at least the original polylines
        original_total = sum(
            section.original_length for section in original_route.sections
        )
        imported_total = sum(
            section.original_length for section in imported_route.sections
        )

        print(f"Original total length: {original_total:.2f}m")
        print(f"Imported total length: {imported_total:.2f}m")

        # Allow some tolerance for round-trip differences
        assert abs(imported_total - original_total) / original_total < 0.1  # Within 10%

    def test_export_with_warnings(self, input_dxf_path, tmp_path) -> None:
        """Test exporting route with warnings."""
        from easycablepulling.io.dxf_writer import export_route_to_dxf

        route = load_route_from_dxf(input_dxf_path)

        # Create some sample warnings
        warnings = [
            "Section AB: Length error 0.3% exceeds maximum 0.2%",
            "Section BC: Bend radius 0.8m is less than minimum 1.0m",
            "Route validation completed with warnings",
        ]

        # Create some sample analysis results
        analysis_results = {
            "max_tension": 15000.0,
            "max_sidewall_pressure": 3500.0,
            "optimal_direction": "Forward",
        }

        output_path = tmp_path / "route_with_warnings.dxf"
        export_route_to_dxf(
            route=route,
            file_path=output_path,
            analysis_results=analysis_results,
            warnings=warnings,
        )

        assert output_path.exists()

        # Verify file can be read back
        reader = DXFReader(output_path)
        reader.load()

        layers = reader.get_layers()
        expected_layers = ["ROUTE_ORIGINAL", "ANNOTATIONS", "JOINTS", "WARNINGS"]

        for expected_layer in expected_layers:
            assert expected_layer in layers, f"Missing layer: {expected_layer}"

        print(f"Export successful with layers: {layers}")
