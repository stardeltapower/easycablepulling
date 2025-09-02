# Developer Guide

## Architecture Overview

EasyCablePulling follows a modular architecture with clear separation of concerns:

```
easycablepulling/
├── core/           # Core models and pipeline
├── io/             # DXF file I/O operations
├── geometry/       # Geometric processing and fitting
├── calculations/   # Cable pulling calculations
├── analysis/       # Statistical analysis
├── reporting/      # Report generation
├── visualization/  # Plotting and visualization
└── cli.py          # Command-line interface
```

## Development Setup

### Environment Setup

```bash
# Clone repository
git clone <repository>
cd easycablepulling

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Quality Tools

```bash
# Run all quality checks
make lint          # flake8, black, isort, mypy
make test          # pytest with coverage
make typecheck     # mypy type checking

# Auto-format code
black easycablepulling/ tests/
isort easycablepulling/ tests/

# Run specific tools
mypy easycablepulling/
flake8 easycablepulling/
pylint easycablepulling/
```

## Core Architecture

### Data Flow

1. **Input**: DXF files with polyline geometry
2. **Parsing**: Extract polylines and convert to Route/Section objects
3. **Geometry Processing**: Fit mathematical primitives (straights/bends)
4. **Splitting**: Optionally split long routes into manageable segments
5. **Calculations**: Compute tensions and pressures for each section
6. **Analysis**: Check against limits and generate recommendations
7. **Output**: Reports, visualizations, and fitted geometry

### Key Components

#### Core Models (`core/models.py`)

**Route Hierarchy:**
```
Route
├── Section (multiple)
    ├── original_polyline: List[Tuple[float, float]]
    └── primitives: List[Primitive]
        ├── Straight(length_m, start_point, end_point)
        └── Bend(radius_m, angle_deg, center_point, ...)
```

**Specifications:**
- `CableSpec`: Physical cable properties and limits
- `DuctSpec`: Duct properties and friction coefficients
- `BendOption`: Available manufactured bend configurations

#### Pipeline (`core/pipeline.py`)

The `CablePullingPipeline` orchestrates the complete analysis workflow:

1. **Input validation**: Check file existence and parameter validity
2. **Route loading**: Parse DXF file into Route object
3. **Geometry processing**: Fit primitives and optionally split route
4. **Cable analysis**: Calculate tensions and pressures
5. **Feasibility check**: Compare against limits with safety factors
6. **Report generation**: Create summary and detailed results

### Adding New Features

#### Adding a New Primitive Type

1. **Define the primitive** in `core/models.py`:

```python
@dataclass
class Spiral(Primitive):
    """Spiral/helical section of cable route."""

    radius_start_m: float
    radius_end_m: float
    pitch_m: float
    turns: float

    def length(self) -> float:
        """Calculate spiral length."""
        # Implementation here
        pass

    def validate(self, cable_spec: CableSpec) -> List[str]:
        """Validate spiral against cable specifications."""
        warnings = []
        # Add spiral-specific validations
        return warnings
```

2. **Add fitting logic** in `geometry/fitter.py`:

```python
def fit_spiral_section(points: List[Tuple[float, float]],
                      tolerance: float) -> Optional[Spiral]:
    """Fit spiral primitive to point sequence."""
    # Implementation here
    pass
```

3. **Add calculation support** in `calculations/`:
   - Update tension calculations for spiral geometry
   - Update pressure calculations for spiral forces
   - Add spiral-specific limit checks

4. **Update visualization** in `visualization/`:
   - Add spiral rendering to plotters
   - Update professional plot templates

#### Adding a New Analysis Feature

1. **Define analysis result model** in `core/models.py`:

```python
@dataclass
class NewAnalysisResult:
    """Results from new analysis feature."""

    section_id: str
    analysis_value: float
    passes_check: bool
    details: Dict[str, Any]
```

2. **Implement analysis function** in `calculations/`:

```python
def perform_new_analysis(section: Section,
                        cable_spec: CableSpec,
                        duct_spec: DuctSpec) -> NewAnalysisResult:
    """Perform new analysis on route section."""
    # Implementation here
    pass
```

3. **Integrate into pipeline** in `core/pipeline.py`:

```python
# Add to run_analysis method
new_analysis_results = []
for section in processed_route.sections:
    analysis_result = perform_new_analysis(section, cable_spec, duct_spec)
    new_analysis_results.append(analysis_result)

# Add to PipelineResult
new_analysis_results=new_analysis_results
```

4. **Add reporting support** in `reporting/`:
   - Update text report generation
   - Add CSV export columns
   - Include in JSON summary

## Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Unit tests for individual modules
├── integration/    # Integration tests for complete workflows
└── data/          # Test data (DXF files, configurations)
```

### Writing Unit Tests

```python
# tests/unit/test_new_feature.py
import pytest
from easycablepulling.feature import NewFeature

class TestNewFeature:
    """Test new feature functionality."""

    def test_basic_functionality(self):
        """Test basic feature operation."""
        feature = NewFeature()
        result = feature.process(test_input)

        assert result.success
        assert result.value > 0

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        feature = NewFeature()

        # Test with invalid input
        with pytest.raises(ValueError):
            feature.process(invalid_input)

        # Test with boundary conditions
        result = feature.process(boundary_input)
        assert result.success or len(result.warnings) > 0

    @pytest.mark.parametrize("input_value,expected", [
        (1.0, 2.0),
        (2.0, 4.0),
        (3.0, 6.0),
    ])
    def test_parameter_variations(self, input_value, expected):
        """Test with different parameter values."""
        feature = NewFeature()
        result = feature.process(input_value)
        assert abs(result.value - expected) < 0.001
```

### Writing Integration Tests

```python
# tests/integration/test_new_workflow.py
class TestNewWorkflow:
    """Test complete workflow with new feature."""

    def test_end_to_end_workflow(self, test_data_dir):
        """Test complete workflow including new feature."""
        pipeline = CablePullingPipeline(enable_new_feature=True)

        result = pipeline.run_analysis(
            test_data_dir / "test_route.dxf",
            cable_spec,
            duct_spec
        )

        assert result.success
        assert hasattr(result, 'new_feature_results')
        assert len(result.new_feature_results) > 0
```

### Test Data Management

Generate synthetic test data for comprehensive testing:

```python
# tests/data/generate_test_cases.py
def generate_test_case(case_name: str, **parameters):
    """Generate synthetic test case."""
    # Create DXF file with specific characteristics
    # Save with descriptive name
    pass

# Generate test cases for specific scenarios
generate_test_case("straight_long", length=2000.0)
generate_test_case("tight_bends", min_radius=300.0)
generate_test_case("s_curve_complex", num_bends=4)
```

## Code Style Guidelines

### Python Style

Follow PEP 8 with these specific guidelines:

```python
# Use type hints consistently
def calculate_tension(length: float, weight: float) -> float:
    """Calculate tension with proper typing."""
    return length * weight * 9.81

# Use dataclasses for data structures
@dataclass
class AnalysisResult:
    """Analysis result with proper structure."""

    value: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

# Handle errors explicitly
def risky_operation(data: List[float]) -> Optional[float]:
    """Operation that might fail."""
    try:
        if not data:
            return None
        return sum(data) / len(data)
    except (TypeError, ZeroDivisionError) as e:
        logger.warning(f"Operation failed: {e}")
        return None

# Use descriptive variable names
def calculate_sidewall_pressure(
    tension_at_exit: float,
    bend_radius_meters: float,
    cable_diameter_mm: float
) -> float:
    """Calculate sidewall pressure with clear naming."""
    # Implementation
    pass
```

### Documentation Standards

#### Docstring Format

```python
def complex_calculation(
    route: Route,
    cable_spec: CableSpec,
    environmental_factors: Dict[str, float],
    **kwargs: Any
) -> CalculationResult:
    """Perform complex cable pulling calculation.

    Calculates cable pulling forces considering route geometry,
    cable specifications, and environmental factors.

    Args:
        route: Cable route with fitted geometry primitives
        cable_spec: Cable physical properties and limits
        environmental_factors: Temperature, slope, etc. factors
        **kwargs: Additional calculation parameters
            - temperature_factor: Temperature adjustment (default 1.0)
            - slope_factor: Slope adjustment (default 0.0)
            - wind_factor: Wind loading factor (default 0.0)

    Returns:
        CalculationResult with tensions, pressures, and feasibility

    Raises:
        ValueError: If route has no sections or invalid cable_spec
        CalculationError: If numerical methods fail to converge

    Example:
        >>> cable_spec = CableSpec(diameter=35.0, weight_per_meter=2.5, ...)
        >>> result = complex_calculation(route, cable_spec, {"temperature": 1.1})
        >>> print(f"Max tension: {result.max_tension:.1f}N")

    Note:
        This function may take significant time for routes with >100 sections.
        Consider using the batch processing interface for large datasets.
    """
    # Implementation
    pass
```

#### Class Documentation

```python
class GeometryProcessor:
    """Process cable route geometry and fit mathematical primitives.

    The GeometryProcessor converts polyline representations of cable routes
    into mathematical primitives (straights, bends, curves) suitable for
    cable pulling analysis. It includes automated fitting algorithms,
    validation, and optional route splitting.

    Attributes:
        tolerance_mm: Geometric fitting tolerance in millimeters
        min_segment_length: Minimum segment length for processing
        max_iterations: Maximum iterations for fitting algorithms

    Example:
        >>> processor = GeometryProcessor(tolerance_mm=1.0)
        >>> result = processor.process_route(route)
        >>> if result.success:
        ...     print(f"Fitted {len(result.fitting_results)} sections")
    """
```

## Performance Guidelines

### Memory Management

```python
# Avoid keeping large objects in memory
def process_large_route(route: Route) -> ProcessingResult:
    """Process large route efficiently."""

    # Process sections individually to manage memory
    section_results = []
    for section in route.sections:
        # Process section
        result = process_section(section)
        section_results.append(result)

        # Clear intermediate data
        del section.temporary_data

    return combine_results(section_results)

# Use generators for large datasets
def analyze_all_sections(route: Route) -> Iterator[SectionResult]:
    """Generator for section analysis to manage memory."""
    for section in route.sections:
        yield analyze_section(section)
```

### Algorithm Efficiency

```python
# Cache expensive calculations
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_calculation(param1: float, param2: float) -> float:
    """Cached expensive calculation."""
    # Complex calculation here
    pass

# Use numpy for vectorized operations
def calculate_tensions_vectorized(positions: np.ndarray,
                                weights: np.ndarray) -> np.ndarray:
    """Vectorized tension calculation."""
    return np.cumsum(weights * positions * 9.81)

# Avoid nested loops where possible
def find_critical_points_efficient(tensions: List[float],
                                 limit: float) -> List[int]:
    """Efficient critical point detection."""
    return [i for i, t in enumerate(tensions) if t > limit]
```

## Extension Points

### Custom Calculation Methods

```python
# Extend calculations by subclassing
class CustomTensionCalculator(BaseTensionCalculator):
    """Custom tension calculation with additional factors."""

    def calculate_section_tension(self, section: Section,
                                 cable_spec: CableSpec) -> SectionTensionAnalysis:
        """Custom tension calculation."""
        # Call base implementation
        base_result = super().calculate_section_tension(section, cable_spec)

        # Add custom factors
        custom_factor = self._calculate_custom_factor(section)

        # Modify results
        modified_tensions = [
            TensionResult(
                position=tr.position,
                tension=tr.tension * custom_factor,
                primitive_index=tr.primitive_index,
                primitive_type=tr.primitive_type
            )
            for tr in base_result.forward_tensions
        ]

        return SectionTensionAnalysis(
            section_id=base_result.section_id,
            forward_tensions=modified_tensions,
            backward_tensions=base_result.backward_tensions,
            max_tension=max(t.tension for t in modified_tensions),
            max_tension_position=base_result.max_tension_position,
            critical_primitive_index=base_result.critical_primitive_index
        )
```

### Custom Geometry Fitting

```python
# Add new fitting algorithms
class CustomFitter(BaseFitter):
    """Custom geometry fitting algorithm."""

    def fit_primitives(self, points: List[Tuple[float, float]],
                      tolerance: float) -> List[Primitive]:
        """Custom primitive fitting algorithm."""
        # Implement custom fitting logic
        pass

    def fit_section(self, section: Section,
                   tolerance: float) -> FittingResult:
        """Fit custom primitives to section."""
        primitives = self.fit_primitives(section.original_polyline, tolerance)

        return FittingResult(
            primitives=primitives,
            total_error=self._calculate_total_error(primitives, section.original_polyline),
            max_error=self._calculate_max_error(primitives, section.original_polyline),
            fitted_points=self._generate_fitted_points(primitives),
            success=True,
            message=f"Fitted {len(primitives)} custom primitives"
        )
```

### Custom Reporters

```python
# Add new report formats
class CustomReporter:
    """Custom report generator."""

    @staticmethod
    def generate_xml_report(result: PipelineResult) -> str:
        """Generate XML format report."""
        xml_elements = []
        xml_elements.append('<?xml version="1.0" encoding="UTF-8"?>')
        xml_elements.append('<cable_analysis>')

        # Add route information
        xml_elements.append(f'  <route name="{result.processed_route.name}">')
        xml_elements.append(f'    <total_length_m>{result.summary["total_length_m"]}</total_length_m>')
        xml_elements.append(f'    <feasible>{result.summary["feasibility"]["overall_feasible"]}</feasible>')
        xml_elements.append('  </route>')

        # Add section details
        xml_elements.append('  <sections>')
        for tension_analysis, limit_result in zip(result.tension_analyses, result.limit_results):
            xml_elements.append(f'    <section id="{tension_analysis.section_id}">')
            xml_elements.append(f'      <max_tension_n>{tension_analysis.max_tension}</max_tension_n>')
            xml_elements.append(f'      <max_pressure_n_per_m>{limit_result.max_pressure}</max_pressure_n_per_m>')
            xml_elements.append(f'      <feasible>{limit_result.passes_all_limits}</feasible>')
            xml_elements.append('    </section>')
        xml_elements.append('  </sections>')

        xml_elements.append('</cable_analysis>')
        return '\n'.join(xml_elements)
```

## Debugging Guidelines

### Logging Configuration

```python
import logging

# Configure logging for development
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('easycablepulling.log'),
        logging.StreamHandler()
    ]
)

# Module-specific loggers
logger = logging.getLogger(__name__)
```

### Debug Utilities

```python
# Add debug output to geometry processing
def debug_fitting_process(section: Section, fitting_result: FittingResult):
    """Debug geometry fitting process."""

    print(f"Section {section.id}:")
    print(f"  Original points: {len(section.original_polyline)}")
    print(f"  Fitted primitives: {len(fitting_result.primitives)}")
    print(f"  Total error: {fitting_result.total_error:.3f}mm")
    print(f"  Max error: {fitting_result.max_error:.3f}mm")

    for i, primitive in enumerate(fitting_result.primitives):
        print(f"  Primitive {i}: {type(primitive).__name__}, length={primitive.length():.1f}m")

# Debug tension calculations
def debug_tension_analysis(analysis: SectionTensionAnalysis):
    """Debug tension analysis results."""

    print(f"Section {analysis.section_id} tension analysis:")
    print(f"  Max tension: {analysis.max_tension:.1f}N at position {analysis.max_tension_position:.1f}m")

    print("  Forward tensions:")
    for tr in analysis.forward_tensions:
        print(f"    {tr.position:.1f}m: {tr.tension:.1f}N ({tr.primitive_type})")
```

### Common Debugging Scenarios

#### Geometry Fitting Issues

```python
# Debug poor fitting results
def investigate_fitting_issues(section: Section, tolerance: float):
    """Investigate geometry fitting problems."""

    points = section.original_polyline

    # Check point density
    distances = []
    for i in range(1, len(points)):
        dx = points[i][0] - points[i-1][0]
        dy = points[i][1] - points[i-1][1]
        dist = math.sqrt(dx**2 + dy**2)
        distances.append(dist)

    print(f"Point spacing: min={min(distances):.1f}mm, max={max(distances):.1f}mm")
    print(f"Total original length: {sum(distances):.1f}mm")

    # Try different fitting approaches
    from easycablepulling.geometry.fitter import AdvancedFitter
    fitter = AdvancedFitter(tolerance_mm=tolerance)

    result = fitter.fit_section(section, tolerance)
    print(f"Fitting result: {result.success}, error={result.total_error:.3f}mm")
```

#### Calculation Validation

```python
# Validate calculation results
def validate_calculation_results(result: PipelineResult):
    """Validate calculation results for physical reasonableness."""

    for tension_analysis in result.tension_analyses:
        # Check for negative tensions
        all_tensions = [tr.tension for tr in tension_analysis.forward_tensions]
        all_tensions.extend([tr.tension for tr in tension_analysis.backward_tensions])

        if any(t < 0 for t in all_tensions):
            print(f"WARNING: Negative tension in section {tension_analysis.section_id}")

        # Check for unreasonable tension values
        if tension_analysis.max_tension > result.cable_spec.max_tension * 3:
            print(f"WARNING: Very high tension in section {tension_analysis.section_id}")

    # Validate pressure calculations
    for limit_result in result.limit_results:
        if limit_result.max_pressure < 0:
            print(f"WARNING: Negative pressure in section {limit_result.section_id}")
```

## Contributing Guidelines

### Submitting Changes

1. **Fork and branch**: Create feature branch from `develop`
2. **Implement changes**: Follow code style and add tests
3. **Run quality checks**: Ensure all linting and tests pass
4. **Update documentation**: Add/update relevant documentation
5. **Submit PR**: Include description of changes and test results

### Code Review Checklist

- [ ] Code follows style guidelines (black, isort, flake8)
- [ ] Type hints are complete and accurate (mypy passes)
- [ ] Unit tests cover new functionality
- [ ] Integration tests verify end-to-end behavior
- [ ] Documentation is updated (docstrings, user guide, API reference)
- [ ] Performance impact is acceptable
- [ ] Backward compatibility is maintained
- [ ] Error handling is comprehensive

### Testing Requirements

- **Unit test coverage**: >90% for new code
- **Integration tests**: Cover main user workflows
- **Performance tests**: Verify acceptable execution time
- **Memory tests**: Check for memory leaks in long-running operations
- **Error handling tests**: Verify graceful error recovery

## Release Process

### Version Management

```python
# Update version in setup.py and __init__.py
__version__ = "0.2.0"

# Follow semantic versioning
# MAJOR.MINOR.PATCH
# MAJOR: Breaking API changes
# MINOR: New features (backward compatible)
# PATCH: Bug fixes
```

### Pre-Release Checklist

1. **All tests pass**: `make test` with 100% success
2. **Documentation current**: All docs reflect current API
3. **Performance benchmarks**: No significant regressions
4. **Examples work**: All example code executes successfully
5. **Dependencies updated**: Requirements reflect current needs

### Release Notes

```markdown
## Version 0.2.0

### New Features
- Added spiral primitive support for helical routes
- Implemented advanced environmental factor calculations
- Added XML export format for analysis results

### Improvements
- 20% faster geometry processing for complex routes
- Better error messages for geometry fitting failures
- Enhanced validation for edge case geometries

### Bug Fixes
- Fixed memory leak in long route processing
- Corrected pressure calculation for trefoil arrangements
- Fixed visualization scaling for very small routes

### Breaking Changes
- `AnalysisResult.tension_data` renamed to `AnalysisResult.tension_analyses`
- `calculate_pressure()` now requires `cable_arrangement` parameter

### Dependencies
- Updated numpy requirement to >=1.24.0
- Added optional plotly dependency for enhanced visualization
```

## Troubleshooting Development Issues

### Common Build Issues

```bash
# Clean build environment
make clean
rm -rf build/ dist/ *.egg-info/

# Reinstall in development mode
pip install -e ".[dev]"

# Update dependencies
pip install -r requirements-dev.txt --upgrade
```

### Test Failures

```bash
# Run specific failing test with debug output
python -m pytest tests/unit/test_module.py::TestClass::test_method -v -s --tb=long

# Run with debugger
python -m pytest tests/unit/test_module.py::TestClass::test_method --pdb

# Check coverage gaps
python -m pytest --cov=easycablepulling --cov-report=html
# Open htmlcov/index.html to see uncovered lines
```

### Type Checking Issues

```bash
# Run mypy with detailed output
mypy easycablepulling/ --show-error-codes --show-error-context

# Check specific module
mypy easycablepulling/core/models.py --disallow-any-expr

# Ignore specific errors (use sparingly)
result = some_function()  # type: ignore[return-value]
```

## Best Practices

### Error Handling

```python
# Specific exception types
class GeometryError(Exception):
    """Geometry processing error."""
    pass

class CalculationError(Exception):
    """Cable calculation error."""
    pass

# Graceful degradation
def robust_analysis(route: Route) -> AnalysisResult:
    """Analysis with fallback strategies."""
    try:
        return advanced_analysis(route)
    except CalculationError:
        logger.warning("Advanced analysis failed, using simplified method")
        return simplified_analysis(route)
    except Exception as e:
        logger.error(f"All analysis methods failed: {e}")
        return AnalysisResult.failed(str(e))
```

### Configuration Management

```python
# Use pydantic for configuration validation
from pydantic import BaseSettings

class AnalysisConfig(BaseSettings):
    """Analysis configuration with validation."""

    geometric_tolerance: float = 1.0
    max_cable_length: float = 500.0
    safety_factor: float = 1.5

    class Config:
        env_prefix = "EASYCABLE_"

# Load from environment or file
config = AnalysisConfig()
pipeline = CablePullingPipeline(**config.dict())
```

### Monitoring and Profiling

```python
# Add timing decorators
import time
from functools import wraps

def timed(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug(f"{func.__name__} took {end-start:.3f}s")
        return result
    return wrapper

@timed
def expensive_operation(data):
    """Operation with timing."""
    # Implementation
    pass

# Memory profiling
def profile_memory_usage(func):
    """Profile memory usage of function."""
    import psutil
    import os

    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        result = func(*args, **kwargs)

        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        logger.debug(f"{func.__name__} memory: {mem_after-mem_before:.1f}MB")

        return result
    return wrapper
```
