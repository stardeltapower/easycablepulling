# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure
- Core modules for geometry, calculations, and I/O
- Basic test suite
- Pre-commit hooks for code quality
- Documentation structure
- **Core data models** (Phase 1 complete):
  - CableSpec with multi-cable arrangement support (single, trefoil, flat)
  - DuctSpec with friction coefficients and bend options
  - Geometric primitives (Straight and Bend classes)
  - Section and Route classes for organizing geometry
  - Comprehensive validation for all models
  - 20 unit tests with 90% coverage for models
- **DXF Import/Export** (Phase 2 complete):
  - DXFReader for loading polylines from DXF files
  - PolylineParser for automatic section identification
  - DXFWriter with organized layer management
  - Support for annotations, joint markers, and analysis results
  - Round-trip import/export with geometry preservation
  - 8 integration tests with real DXF data
  - Tested with 6.1km route containing 13 polylines
- **Geometry Processing** (Phase 3 complete):
  - Diameter-based bend classification (natural vs manufactured)
  - Natural bend threshold: 20-25× duct diameter (research-based)
  - Advanced polynomial curve fitting for complex snaking routes
  - Recursive geometry fitting with fallback strategies
  - Path following improvements achieving 93% within 1m deviation
  - Length preservation with 0.10% overall error
  - Arc generation with proper control points and angles
  - Standard duct bend library (45°/90° at various radii)
  - Automatic straight segment rejoining for construction efficiency
  - Comprehensive deviation analysis and validation tools

### Changed
- Updated test fixtures to use actual model objects instead of dictionaries
- Bend model now includes control_points and start/end angles for accurate arc generation
- Improved geometry fitting to prioritize path following over simplification

### Deprecated
- None

### Removed
- None

### Fixed
- None

### Security
- None

## [0.1.0] - 2024-01-XX

### Added
- Initial release
- Basic cable pulling calculations
- DXF import/export functionality
- Geometry fitting algorithms
- Command-line interface
