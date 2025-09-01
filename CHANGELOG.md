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

### Changed
- Updated test fixtures to use actual model objects instead of dictionaries

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
