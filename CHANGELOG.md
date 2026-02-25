# Changelog

All notable changes to this project will be documented in this file.

## [0.2.1] - 2026-02-25

### Fixed
- Fixed `TopologyException` crash on thin/elongated polygons by validating smoothed variants before union
- Fixed crash when hole subtraction splits a polygon into a MultiPolygon (e.g. tiny holes relative to segment length)

## [0.2.0] - 2026-02-25

### Fixed
- `smooth_iterations=0` now returns the original input unchanged instead of running the geometry through segmentize/simplify pipeline without smoothing

## [0.1.0] - 2025-11-25

### Added
- Initial public release
- Core smoothing functionality using Chaikin's corner-cutting algorithm
- Support for all Shapely geometry types (Polygon, LineString, MultiPolygon, etc.)
- Automatic segment length detection
- Parallel processing support
- Area preservation for polygons