"""Integration tests for DXF file loading."""


class TestDXFLoading:
    """Test DXF file loading functionality."""

    def test_input_dxf_exists(self, input_dxf_path) -> None:
        """Test that the default input.dxf file exists."""
        assert input_dxf_path.exists(), f"Input DXF file not found at {input_dxf_path}"
        assert input_dxf_path.suffix == ".dxf"
        assert input_dxf_path.name == "input.dxf"

    def test_input_dxf_readable(self, input_dxf_path) -> None:
        """Test that the input.dxf file can be opened."""
        # Basic readability test
        with open(input_dxf_path, "r") as f:
            first_line = f.readline().strip()
            # DXF files typically start with a version indicator
            assert first_line in ["0", "999"], f"Unexpected first line: {first_line}"
