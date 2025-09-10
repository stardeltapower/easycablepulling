# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed
- **Critical tension calculation fix**: Corrected unrealistic reverse tension calculations that were showing 10-50x higher values than forward tensions
- **Sidewall pressure physics**: Fixed sidewall pressures to be direction-dependent using actual tension at each bend location (P = T/R)
- **Reverse tension methodology**: Implemented proper symmetrical calculation using same methodology as forward with flipped geometry
- **Cable weight configuration**: Corrected default cable weight from 2.5kg/m to 1.5kg/m as specified
- **CSV reverse tension ordering**: Fixed reverse tensions to display in correct descending order (high to low)
- **Character encoding**: Replaced problematic degree symbol (°) with "deg" to avoid UTF-8 display issues

### Added
- **Reverse sidewall pressure column**: Added separate column for reverse direction sidewall pressures in CSV exports
- **Direction-dependent pressure calculations**: Sidewall pressures now calculated separately for forward and reverse pulling
- **Realistic tension ratios**: Final tensions now show 14/16 sections with reverse < forward as expected in real cable pulling

### Changed
- **CSV format**: Extended section CSV files with additional "Reverse Sidewall (N/m)" column
- **Tension physics**: Reverse tensions now use same initial values and methodology as forward, resulting in realistic 0.02-1.00 ratios
- **Documentation**: Updated README with improved calculation descriptions and corrected default parameters

### Technical Details
- **Forward/reverse ratio range**: 0.02 to 1.00 (realistic cable pulling ranges)
- **Sidewall pressure validation**: Confirmed with Elek calculator reference that pressures depend on actual tension
- **Physics compliance**: All calculations now follow proper cable pulling physics with symmetrical methodology

## [Previous Versions]
- Phase 7: Testing and Documentation
- Phase 6: Professional Reporting and Visualization
- Phase 5: Analysis Pipeline and CLI Enhancement
- Phase 3: Enhanced Geometry Processing
