# Low-Angle Basketball Shot Detection Adaptations

## Overview
This document outlines the key adaptations made to the basketball shot detection system to work with low-angle court perspectives, where the camera is positioned at court level looking horizontally across the court.

## Key Differences from Original Implementation

### 1. Viewing Angle Challenges
- **Original**: High-angle or overhead view with steep ball trajectories
- **Low-Angle**: Horizontal court-level view with compressed depth perception
- **Impact**: Ball and rim appear much smaller, trajectories are more horizontal

### 2. Detection Threshold Adjustments

#### Ball Detection (`shot_detector_low_angle.py` lines 77-80)
```python
# Reduced from 0.3 to 0.25, and from 0.15 to 0.1 for near-hoop detection
if (conf > .25 or (in_hoop_region(center, self.hoop_pos) and conf > 0.1)) and current_class == "Basketball":
```

#### Rim Detection (`shot_detector_low_angle.py` lines 83-86)
```python
# Reduced from 0.5 to 0.35 due to smaller rim appearance
if conf > .35 and current_class == "Basketball Hoop":
```

### 3. Shot Detection Logic Adaptations

#### Up Detection (`utils_low_angle.py` lines 58-89)
- **Original**: Detected ball in backboard area above rim
- **Adapted**: Detects ball approaching rim area from any direction
- **Key Changes**:
  - Larger detection zone (4x rim dimensions vs 2x)
  - Movement-based detection (ball moving toward rim)
  - More flexible spatial positioning

#### Down Detection (`utils_low_angle.py` lines 31-55)
- **Original**: Detected ball below net level
- **Adapted**: Detects ball moving away from rim area
- **Key Changes**:
  - Distance-based detection from rim center
  - 3x rim width as buffer zone
  - Works for shots from any angle

### 4. Scoring Algorithm (`utils_low_angle.py` lines 16-62)

#### Original Approach
- Used trajectory intersection with rim height
- Relied on vertical ball movement patterns

#### Low-Angle Approach
- **Direct Position Check**: Analyzes if ball passed through scoring zone
- **Trajectory Analysis**: Uses point-to-line distance calculations
- **Generous Scoring Zone**: 80% of detected rim dimensions
- **Multi-Position Analysis**: Checks last 10 ball positions for rim passage

### 5. Enhanced Object Tracking

#### Ball Position Cleaning (`utils_low_angle.py` lines 124-153)
- Reduced maximum movement distance (3x vs 4x diameter)
- Stricter aspect ratio validation (1.2 vs 1.4)
- Longer tracking history (40 vs 30 frames)

#### Rim Position Cleaning (`utils_low_angle.py` lines 156-183)
- Stricter movement constraints (0.3x vs 0.5x diameter)
- Tighter aspect ratio for circular rim detection (1.1 vs 1.3)
- Extended tracking history (35 vs 25 positions)

### 6. Visual Enhancements

#### Detection Visualization
- Different colors for ball (orange) and rim (green) detection boxes
- Ball trail with intensity-based coloring
- Rim detection zone overlay for debugging
- Larger rim center marker for visibility

#### UI Improvements
- Background rectangles for text visibility
- Shooting percentage display
- Real-time detection status
- Enhanced shot result overlays

## Usage Instructions

### 1. File Structure
```
shot_detector_low_angle.py    # Main detection class for low-angle view
utils_low_angle.py           # Adapted utility functions
shot_detector.py             # Original implementation (unchanged)
utils.py                     # Original utility functions (unchanged)
```

### 2. Running the Low-Angle Detector
```python
from shot_detector_low_angle import ShotDetectorLowAngle

# Initialize with your trained model and video
detector = ShotDetectorLowAngle(
    model_path="1200images-yolon11/best.pt",  # Your trained model
    video_path="your_video.mp4"               # Your test video
)
```

### 3. Key Parameters to Tune

#### Confidence Thresholds
- Ball detection: `conf > .25` (line 78 in shot_detector_low_angle.py)
- Rim detection: `conf > .35` (line 83 in shot_detector_low_angle.py)

#### Detection Zones
- Up zone: `rim_width * 4` (line 80 in utils_low_angle.py)
- Down zone: `rim_width * 3` (line 54 in utils_low_angle.py)
- Scoring zone: `rim_width * 0.8` (line 26 in utils_low_angle.py)

#### Timing Parameters
- Detection check interval: Every 15 frames (line 100 in shot_detector_low_angle.py)
- Minimum shot duration: 5 frames (line 102 in shot_detector_low_angle.py)
- State timeout: 60 frames for up, 45 frames for down (lines 115-121)

## Recommended Fine-Tuning

### For Different Court Angles
1. **Adjust detection zones** based on rim size in your footage
2. **Modify confidence thresholds** based on model performance
3. **Tune timeout values** based on typical shot durations

### For Different Lighting Conditions
1. **Lower confidence thresholds** for dimmer environments
2. **Increase detection zone sizes** for challenging visibility
3. **Adjust cleaning parameters** for noisy detections

### For Different Shot Types
1. **Expand up detection zone** for long-range shots
2. **Adjust scoring zone size** for different rim perspectives
3. **Modify trajectory analysis window** for varying shot speeds

## Performance Considerations

1. **Model Selection**: Use the model trained specifically for your camera angle
2. **Processing Speed**: Lower confidence thresholds may increase false positives
3. **Accuracy vs Speed**: Larger detection zones improve accuracy but may slow processing
4. **Memory Usage**: Longer tracking history improves accuracy but uses more memory

## Debugging Features

- Visual detection zones displayed on frame
- Real-time detection status in bottom-left corner
- Ball trail visualization with recency-based intensity
- Enhanced shot result feedback with center positioning
