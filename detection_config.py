# Configuration file for basketball shot detection parameters
# Adjust these values based on your specific footage and requirements

class DetectionConfig:
    """Configuration class for shot detection parameters"""
    
    # Confidence thresholds
    BALL_CONFIDENCE_THRESHOLD = 0.2        # Minimum confidence for ball detection
    BALL_NEAR_HOOP_CONFIDENCE = 0.08       # Lower threshold when ball is near hoop
    HOOP_CONFIDENCE_THRESHOLD = 0.25       # Minimum confidence for rim detection
    
    # Detection zones (multipliers of rim dimensions)
    UP_ZONE_MULTIPLIER = 5.0               # Size of approach detection zone
    DOWN_ZONE_MULTIPLIER = 3.0             # Size of departure detection zone
    HOOP_REGION_MULTIPLIER = 2.0           # Size of "near hoop" region
    
    # Scoring parameters
    SCORE_ZONE_MULTIPLIER = 1.0            # Size of scoring zone
    TRAJECTORY_CHECK_POSITIONS = 15        # Number of ball positions to check for scoring
    TRAJECTORY_SEGMENTS = 8                # Number of trajectory segments to analyze
    MIN_NEAR_RIM_FRAMES = 3                # Minimum frames ball should be near rim for enhanced make detection
    
    # Timing parameters
    DETECTION_CHECK_INTERVAL = 10          # Frames between detection checks
    MIN_SHOT_DURATION = 3                  # Minimum frames between up and down detection
    UP_TIMEOUT_FRAMES = 90                 # Frames before up state times out
    DOWN_TIMEOUT_FRAMES = 60               # Frames before down state times out
    
    # Ball tracking parameters
    MAX_BALL_MOVEMENT_MULTIPLIER = 3.0     # Maximum ball movement (multiplier of ball size)
    BALL_ASPECT_RATIO_TOLERANCE = 1.2      # Maximum aspect ratio deviation
    BALL_TRACKING_HISTORY = 40             # Frames to keep in ball position history
    
    # Rim tracking parameters  
    MAX_RIM_MOVEMENT_MULTIPLIER = 0.3      # Maximum rim movement (multiplier of rim size)
    RIM_ASPECT_RATIO_TOLERANCE = 1.1       # Maximum aspect ratio deviation for rim
    RIM_TRACKING_HISTORY = 35              # Number of rim positions to keep
    
    # Visual settings
    DEBUG_MODE = True                      # Enable debug visualization
    FADE_FRAMES = 30                       # Frames for result overlay fade
    
    @classmethod
    def get_config_dict(cls):
        """Return configuration as a dictionary"""
        return {attr: getattr(cls, attr) for attr in dir(cls) 
                if not attr.startswith('_') and not callable(getattr(cls, attr))}
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=== Shot Detection Configuration ===")
        config = cls.get_config_dict()
        for key, value in config.items():
            print(f"{key}: {value}")
        print("=" * 40)

# Easy adjustment presets
class ConfigPresets:
    """Predefined configuration presets for different scenarios"""
    
    @staticmethod
    def high_sensitivity():
        """Configuration for better shot detection sensitivity"""
        DetectionConfig.BALL_CONFIDENCE_THRESHOLD = 0.15
        DetectionConfig.BALL_NEAR_HOOP_CONFIDENCE = 0.05
        DetectionConfig.HOOP_CONFIDENCE_THRESHOLD = 0.2
        DetectionConfig.UP_ZONE_MULTIPLIER = 6.0
        DetectionConfig.SCORE_ZONE_MULTIPLIER = 1.2
        DetectionConfig.MIN_NEAR_RIM_FRAMES = 2
    
    @staticmethod
    def high_accuracy():
        """Configuration for more precise detection (fewer false positives)"""
        DetectionConfig.BALL_CONFIDENCE_THRESHOLD = 0.3
        DetectionConfig.BALL_NEAR_HOOP_CONFIDENCE = 0.15
        DetectionConfig.HOOP_CONFIDENCE_THRESHOLD = 0.4
        DetectionConfig.UP_ZONE_MULTIPLIER = 4.0
        DetectionConfig.SCORE_ZONE_MULTIPLIER = 0.8
        DetectionConfig.MIN_NEAR_RIM_FRAMES = 4
        DetectionConfig.MIN_SHOT_DURATION = 5
    
    @staticmethod
    def fast_shots():
        """Configuration for fast-paced gameplay"""
        DetectionConfig.DETECTION_CHECK_INTERVAL = 5
        DetectionConfig.MIN_SHOT_DURATION = 2
        DetectionConfig.UP_TIMEOUT_FRAMES = 60
        DetectionConfig.DOWN_TIMEOUT_FRAMES = 45
    
    @staticmethod
    def slow_shots():
        """Configuration for slower, more deliberate shots"""
        DetectionConfig.DETECTION_CHECK_INTERVAL = 15
        DetectionConfig.MIN_SHOT_DURATION = 8
        DetectionConfig.UP_TIMEOUT_FRAMES = 120
        DetectionConfig.DOWN_TIMEOUT_FRAMES = 90

# Usage examples:
# ConfigPresets.high_sensitivity()    # Apply high sensitivity preset
# DetectionConfig.print_config()      # Print current configuration
# DetectionConfig.DEBUG_MODE = False  # Disable debug mode
