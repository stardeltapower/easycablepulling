# TODO List for EasyCablePulling

## Current Status
- ✅ Library modified to handle cable arrangements (trefoil, flat, single) internally
- ✅ A4 paper sizing implemented for matplotlib exports (300 DPI)
- ✅ Route optimization module created for directional analysis
- ⚠️ Some function signatures need fixing for compatibility

## Immediate Fixes Needed

### 1. Fix `calculate_bend_tension()` Function Signature
**File**: `easycablepulling/calculations/tension.py`
**Issue**: Function expects `bend_angle` not `bend_angle_rad`
**Fix**: Update function call in `route_optimizer.py` line 231

### 2. Directional Sidewall Pressure Calculation
**Status**: Partially implemented
**Issue**: Current library doesn't separate forward/reverse sidewall pressures in results
**Fix**: Need to calculate sidewall pressure based on actual tension at each bend for each direction

### 3. Complete Integration Testing
- [ ] Test `run.py` with 500m section splits
- [ ] Test `run_optimized.py` with automatic optimization
- [ ] Verify trefoil calculations are correct

## Features Completed

### Cable Arrangement Support
- ✅ Added `cable_arrangement` parameter to AnalysisConfig
- ✅ Library now calculates bundle diameter internally (2.154× for trefoil)
- ✅ Total weight calculated automatically (n × individual weight)

### Visualization Updates
- ✅ Matplotlib figures auto-detect orientation (landscape/portrait)
- ✅ A4 paper sizing without stretching
- ✅ 300 DPI for all exports
- ✅ Reduced padding to maximize A4 space usage

### Route Optimization
- ✅ Created `RouteOptimizer` class for automatic section splitting
- ✅ Separate analysis for forward and reverse pulling directions
- ✅ Automatic splitting to stay within limits (80% default safety margin)
- ✅ Section length balancing for even distribution

### Enhanced Reporting
- ✅ Section-by-section directional analysis table
- ✅ Individual pass/fail indicators for each metric
- ✅ Length statistics to correlate with tensions
- ✅ Critical section identification with lengths

## Next Steps

1. **Fix Function Compatibility**
   - Update tension calculation function calls
   - Ensure all parameters match expected signatures

2. **Complete Directional Analysis**
   - Implement proper bend-by-bend sidewall pressure calculation
   - Store separate forward/reverse sidewall values in results

3. **Testing & Validation**
   - Run full analysis on Midlands project
   - Verify optimization produces feasible pulling plans
   - Compare forward vs reverse strategies

4. **Documentation Updates**
   - Update USAGE_GUIDE.md with new features
   - Add examples for trefoil configuration
   - Document optimization module usage

## Known Limitations

1. **Sidewall Pressure**: Currently using approximations in `run.py`. The `run_optimized.py` calculates more accurately but needs function signature fix.

2. **Cable Arrangements**: While the model supports arrangements, the pipeline doesn't fully expose this through all interfaces yet.

3. **Visualization**: Plotly visualizations may need updates to handle trefoil bundle diameter correctly.

## Future Enhancements

- [ ] Add support for multiple cable types in same route
- [ ] Implement lubrication zones (partial lubrication)
- [ ] Add cost optimization (balancing number of pits vs pulling difficulty)
- [ ] Create interactive route planning tool
- [ ] Add support for vertical sections and slopes

## Notes

- Trefoil formation uses factor of 2.154 for bundle diameter
- Default safety margin is 80% of limits (20% reserve)
- Sidewall pressure is often the limiting factor, not pulling tension
- Forward and reverse pulling can have significantly different requirements