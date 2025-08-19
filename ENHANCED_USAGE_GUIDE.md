# Enhanced Basketball Shot Detection - Usage Guide

## Quick Start

The enhanced shot detection system has been specifically optimized for your low-angle court footage with improved sensitivity and debugging capabilities.

### 1. Basic Usage

```python
from shot_detector_low_angle import ShotDetectorLowAngle

# Run with your trained model and video
detector = ShotDetectorLowAngle(
    model_path="300images-yolom11/best.pt",
    video_path="sample180s_video-1.mp4"
)
```

### 2. Configuration Presets

The system includes preset configurations for different scenarios:

```python
from detection_config import DetectionConfig, ConfigPresets

# For maximum sensitivity (catches more shots, may have false positives)
ConfigPresets.high_sensitivity()

# For higher accuracy (fewer false positives, may miss some shots)
ConfigPresets.high_accuracy()

# For fast-paced games
ConfigPresets.fast_shots()

# For slower, deliberate shooting
ConfigPresets.slow_shots()

# Then run the detector
detector = ShotDetectorLowAngle(model_path="...", video_path="...")
```

### 3. Manual Configuration

You can also manually adjust parameters:

```python
from detection_config import DetectionConfig

# Adjust confidence thresholds
DetectionConfig.BALL_CONFIDENCE_THRESHOLD = 0.15      # Lower = more sensitive
DetectionConfig.HOOP_CONFIDENCE_THRESHOLD = 0.2       # Lower = more sensitive

# Adjust detection zones
DetectionConfig.UP_ZONE_MULTIPLIER = 6.0              # Larger = wider detection area
DetectionConfig.SCORE_ZONE_MULTIPLIER = 1.2           # Larger = more generous scoring

# Timing adjustments
DetectionConfig.DETECTION_CHECK_INTERVAL = 8          # Lower = faster response
DetectionConfig.MIN_SHOT_DURATION = 2                 # Lower = catches quicker shots

# Enable/disable debug mode
DetectionConfig.DEBUG_MODE = True                     # Shows detailed info on screen
```

## Key Improvements Made

### 1. Enhanced Shot Detection
- **Lower Confidence Thresholds**: Ball detection threshold reduced to 0.15, rim detection to 0.2
- **Larger Detection Zones**: 6x rim size for approach detection (vs 4x in original)
- **Multiple Make Detection Methods**: 
  - Direct position checking
  - Trajectory analysis
  - Ball "disappearance" detection near rim
  - Enhanced make detection using ball dwell time

### 2. Better Tracking
- **Improved Ball Tracking**: Stricter movement validation, longer history (40 frames)
- **Enhanced Rim Tracking**: More stable rim position with 35 frame history
- **Noise Reduction**: Better filtering of erratic detections

### 3. Debug Visualization
When `DEBUG_MODE = True`, you'll see:
- Real-time detection state information
- Ball and rim confidence scores
- Detection zone overlays
- Frame counts and timeouts
- Enhanced make detection criteria results

### 4. Adaptive Timing
- **Faster Response**: Checks every 10 frames (vs 15 in original)
- **Flexible Timeouts**: 90 frames for up detection, 60 for down detection
- **Configurable Intervals**: All timing can be adjusted via configuration

## Troubleshooting Common Issues

### 1. "Not detecting shots consistently"
```python
# Try high sensitivity preset
ConfigPresets.high_sensitivity()

# Or manually increase detection zones
DetectionConfig.UP_ZONE_MULTIPLIER = 7.0
DetectionConfig.DOWN_ZONE_MULTIPLIER = 4.0
```

### 2. "Missing made shots"
```python
# Increase scoring zone and reduce requirements
DetectionConfig.SCORE_ZONE_MULTIPLIER = 1.3
DetectionConfig.MIN_NEAR_RIM_FRAMES = 2

# Lower ball confidence near rim
DetectionConfig.BALL_NEAR_HOOP_CONFIDENCE = 0.03
```

### 3. "Too many false positives"
```python
# Use high accuracy preset
ConfigPresets.high_accuracy()

# Or increase minimum shot duration
DetectionConfig.MIN_SHOT_DURATION = 5
```

### 4. "Detection is too slow"
```python
# Use fast shots preset
ConfigPresets.fast_shots()

# Or manually adjust timing
DetectionConfig.DETECTION_CHECK_INTERVAL = 5
DetectionConfig.MIN_SHOT_DURATION = 2
```

## Advanced Usage

### Custom Configuration for Your Footage
1. Run with `DEBUG_MODE = True` first to understand detection patterns
2. Watch the debug output to see confidence scores and detection states
3. Adjust parameters based on what you observe:
   - Low ball confidence scores → reduce `BALL_CONFIDENCE_THRESHOLD`
   - Shots not triggering → increase `UP_ZONE_MULTIPLIER`
   - Makes being missed → increase `SCORE_ZONE_MULTIPLIER`

### Performance Optimization
```python
# For better performance (less accurate)
DetectionConfig.DEBUG_MODE = False
DetectionConfig.DETECTION_CHECK_INTERVAL = 15
DetectionConfig.BALL_TRACKING_HISTORY = 25

# For better accuracy (slower)
DetectionConfig.DETECTION_CHECK_INTERVAL = 5
DetectionConfig.TRAJECTORY_CHECK_POSITIONS = 20
DetectionConfig.BALL_TRACKING_HISTORY = 50
```

## Configuration Reference

### Critical Parameters for Your Use Case
- `BALL_CONFIDENCE_THRESHOLD`: Start with 0.15, lower if missing ball detections
- `HOOP_CONFIDENCE_THRESHOLD`: Start with 0.2, lower if missing rim detections
- `UP_ZONE_MULTIPLIER`: Start with 6.0, increase if missing shot attempts
- `SCORE_ZONE_MULTIPLIER`: Start with 1.2, increase if missing makes
- `MIN_NEAR_RIM_FRAMES`: Start with 2, increase for more conservative make detection

### Debug Commands
```python
# Print current configuration
DetectionConfig.print_config()

# View all available parameters
print(DetectionConfig.get_config_dict())
```

## Expected Behavior

With the enhanced system:
1. **More Sensitive**: Should catch shots that were previously missed
2. **Better Make Detection**: Multiple algorithms working together to detect successful shots
3. **Visual Feedback**: Debug mode shows exactly what the system is "thinking"
4. **Configurable**: Easy to adjust for your specific footage characteristics

The system is now pre-configured with high sensitivity settings that should work well for your low-angle footage. Monitor the debug output and adjust parameters as needed for optimal performance.
