# Basketball Shot Detector/Tracker - Adapted for Low-Angle Court View
# Based on original by Avi Shah - July 2023

from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from utils_low_angle import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos, get_device
from detection_config import DetectionConfig, ConfigPresets


class ShotDetectorLowAngle:
    def __init__(self, model_path="1200images-yolon11/best.pt", video_path="sample180s_video-2.mp4", save_output=True):
        # Load the YOLO model - using the trained model for this angle
        self.overlay_text = "Waiting..."
        self.model = YOLO(model_path)
        
        # Uncomment this line to accelerate inference. Note that this may cause errors in some setups.
        # self.model.half()
        
        self.class_names = ['Basketball', 'Basketball Hoop']
        self.device = get_device()
        
        # Video input and output setup
        self.video_path = video_path
        self.save_output = save_output
        self.cap = cv2.VideoCapture(video_path)
        
        # Setup video writer for saving analyzed footage
        if self.save_output:
            # Get video properties
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Input video: {video_path}")
            print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
            
            # Create output filename
            import os
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            self.output_path = f"{video_name}_analyzed.mp4"
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            print(f"Output will be saved as: {self.output_path}")
        else:
            self.video_writer = None

        self.ball_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
        self.hoop_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)

        self.frame_count = 0
        self.frame = None

        self.makes = 0
        self.attempts = 0

        # Used to detect shots (approach and departure from rim)
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0
        
        # Track shot state for better detection
        self.shot_in_progress = False
        self.frames_since_up = 0
        self.frames_since_down = 0
        
        # Enhanced tracking for better detection
        self.ball_near_rim_count = 0
        self.max_ball_near_rim = 0
        self.recent_ball_positions = []  # For trajectory analysis
        
        # Store video properties for annotations
        self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS)) if self.cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        
        # Apply high sensitivity preset for better detection
        ConfigPresets.high_sensitivity()
        
        # Load configuration
        self.debug_mode = DetectionConfig.DEBUG_MODE
        self.last_detection_frame = 0

        # Used for green and red colors after make/miss
        self.fade_frames = DetectionConfig.FADE_FRAMES
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)
        
        # Print configuration if in debug mode
        if self.debug_mode:
            DetectionConfig.print_config()

        self.run()

    def run(self):
        while True:
            ret, self.frame = self.cap.read()

            if not ret:
                # End of the video or an error occurred
                break

            results = self.model(self.frame, stream=True, device=self.device)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # Class Name
                    cls = int(box.cls[0])
                    current_class = self.class_names[cls]

                    center = (int(x1 + w / 2), int(y1 + h / 2))

                    # Use configurable confidence thresholds
                    ball_threshold = DetectionConfig.BALL_CONFIDENCE_THRESHOLD
                    ball_near_hoop_threshold = DetectionConfig.BALL_NEAR_HOOP_CONFIDENCE
                    
                    if (conf > ball_threshold or (in_hoop_region(center, self.hoop_pos) and conf > ball_near_hoop_threshold)) and current_class == "Basketball":
                        self.ball_pos.append((center, self.frame_count, w, h, conf))
                        # Different color for ball detection in low-angle view
                        cvzone.cornerRect(self.frame, (x1, y1, w, h), colorR=(255, 100, 0))
                        
                        # Track ball near rim for enhanced detection
                        if len(self.hoop_pos) > 0 and in_hoop_region(center, self.hoop_pos):
                            self.ball_near_rim_count += 1
                            self.max_ball_near_rim = max(self.max_ball_near_rim, self.ball_near_rim_count)
                        else:
                            self.ball_near_rim_count = 0
                        
                        # Debug: Show ball detection confidence
                        if self.debug_mode:
                            cv2.putText(self.frame, f"Ball: {conf:.2f}", (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)

                    # Use configurable confidence threshold for rim detection
                    if conf > DetectionConfig.HOOP_CONFIDENCE_THRESHOLD and current_class == "Basketball Hoop":
                        self.hoop_pos.append((center, self.frame_count, w, h, conf))
                        # Different color for rim detection
                        cvzone.cornerRect(self.frame, (x1, y1, w, h), colorR=(0, 255, 100))
                        
                        # Debug: Show rim detection confidence
                        if self.debug_mode:
                            cv2.putText(self.frame, f"Rim: {conf:.2f}", (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 100), 1)

            self.clean_motion()
            self.shot_detection()
            self.add_vlm_annotations()  # Add comprehensive annotations for VLM
            self.display_score()
            self.frame_count += 1

            # Save frame to output video
            if self.save_output and self.video_writer is not None:
                self.video_writer.write(self.frame)

            # Show frame (optional - comment out for faster processing)
            cv2.imshow('Frame - Low Angle View', self.frame)

            # Close if 'q' is clicked
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        self.cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"\nAnalyzed video saved as: {self.output_path}")
            print(f"Video contains {self.frame_count} frames with comprehensive annotations")
            print("You can now feed this video to Gemini 2.5 Pro for advanced analysis!")
        cv2.destroyAllWindows()

    def clean_motion(self):
        # Clean and display ball motion
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        for i in range(0, len(self.ball_pos)):
            # Draw ball trail with different colors based on recency
            color_intensity = min(255, 50 + (len(self.ball_pos) - i) * 10)
            cv2.circle(self.frame, self.ball_pos[i][0], 3, (0, 0, color_intensity), 2)

        # Clean hoop motion and display current hoop center
        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            # Draw rim center with larger circle for visibility
            cv2.circle(self.frame, self.hoop_pos[-1][0], 5, (0, 255, 255), 3)
            
            # Draw rim detection zone for debugging
            rim_center = self.hoop_pos[-1][0]
            rim_width = self.hoop_pos[-1][2]
            rim_height = self.hoop_pos[-1][3]
            zone_width = int(rim_width * 2)
            zone_height = int(rim_height * 2)
            
            # Draw detection zone rectangle
            cv2.rectangle(self.frame, 
                         (rim_center[0] - zone_width//2, rim_center[1] - zone_height//2),
                         (rim_center[0] + zone_width//2, rim_center[1] + zone_height//2),
                         (100, 100, 100), 1)

    def shot_detection(self):
        if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
            # Enhanced shot detection for low-angle view
            
            # Track frames since detection states for timeout handling
            if self.up:
                self.frames_since_up += 1
            if self.down:
                self.frames_since_down += 1
            
            # Debug information
            current_up = detect_up(self.ball_pos, self.hoop_pos)
            current_down = detect_down(self.ball_pos, self.hoop_pos)
            
            if self.debug_mode and (current_up or current_down or self.up or self.down):
                debug_text = f"Up: {current_up}, Down: {current_down}, State: {self.up}/{self.down}"
                cv2.putText(self.frame, debug_text, (10, self.frame.shape[0] - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            # Detecting when ball approaches rim area
            if not self.up:
                self.up = current_up
                if self.up:
                    self.up_frame = self.ball_pos[-1][1]
                    self.frames_since_up = 0
                    self.shot_in_progress = True
                    self.max_ball_near_rim = 0  # Reset for new shot
                    if self.debug_mode:
                        print(f"Frame {self.frame_count}: UP detected!")

            # Detecting when ball leaves rim area
            if self.up and not self.down:
                self.down = current_down
                if self.down:
                    self.down_frame = self.ball_pos[-1][1]
                    self.frames_since_down = 0
                    if self.debug_mode:
                        print(f"Frame {self.frame_count}: DOWN detected!")

            # Process shot attempt with enhanced logic
            if self.frame_count % DetectionConfig.DETECTION_CHECK_INTERVAL == 0:
                if self.up and self.down and self.up_frame < self.down_frame:
                    # Ensure minimum time between up and down detection
                    frame_difference = self.down_frame - self.up_frame
                    if frame_difference >= DetectionConfig.MIN_SHOT_DURATION:
                        self.attempts += 1
                        shot_made = score(self.ball_pos, self.hoop_pos)
                        
                        # Enhanced make detection using multiple criteria
                        enhanced_make = self.enhanced_make_detection()
                        final_make = shot_made or enhanced_make
                        
                        if self.debug_mode:
                            print(f"Shot attempt {self.attempts}: Standard score: {shot_made}, Enhanced: {enhanced_make}, Final: {final_make}")
                            print(f"Max ball near rim: {self.max_ball_near_rim}")
                        
                        # Reset states
                        self.up = False
                        self.down = False
                        self.shot_in_progress = False
                        self.frames_since_up = 0
                        self.frames_since_down = 0
                        self.last_detection_frame = self.frame_count

                        # Score detection with enhanced logic
                        if final_make:
                            self.makes += 1
                            self.overlay_color = (0, 255, 0)  # Green for make
                            self.overlay_text = "MAKE!"
                            self.fade_counter = self.fade_frames
                        else:
                            self.overlay_color = (0, 0, 255)  # Red for miss
                            self.overlay_text = "MISS"
                            self.fade_counter = self.fade_frames
                
                # Reset detection states if they've been active too long (timeout)
                if self.frames_since_up > DetectionConfig.UP_TIMEOUT_FRAMES:
                    if self.debug_mode:
                        print(f"Frame {self.frame_count}: UP timeout reset")
                    self.up = False
                    self.frames_since_up = 0
                    self.shot_in_progress = False
                    
                if self.frames_since_down > DetectionConfig.DOWN_TIMEOUT_FRAMES:
                    if self.debug_mode:
                        print(f"Frame {self.frame_count}: DOWN timeout reset")
                    self.down = False
                    self.frames_since_down = 0

    def enhanced_make_detection(self):
        """
        Enhanced make detection using multiple criteria for low-angle view
        """
        if len(self.ball_pos) < 3 or len(self.hoop_pos) < 1:
            return False
        
        rim_center = self.hoop_pos[-1][0]
        rim_width = self.hoop_pos[-1][2]
        rim_height = self.hoop_pos[-1][3]
        
        # Criteria 1: Ball spent significant time near rim
        if self.max_ball_near_rim >= DetectionConfig.MIN_NEAR_RIM_FRAMES:
            return True
        
        # Criteria 2: Ball trajectory passed very close to rim center
        close_pass_threshold = min(rim_width, rim_height) * 0.6
        for ball_data in self.ball_pos[-10:]:  # Check last 10 positions
            ball_pos = ball_data[0]
            distance_to_rim = math.sqrt(
                (ball_pos[0] - rim_center[0]) ** 2 + 
                (ball_pos[1] - rim_center[1]) ** 2
            )
            if distance_to_rim <= close_pass_threshold:
                return True
        
        # Criteria 3: Ball showed characteristic "dip" motion near rim
        if len(self.ball_pos) >= 5:
            recent_y_positions = [pos[0][1] for pos in self.ball_pos[-5:]]
            # Check for downward then upward motion (ball going through rim)
            min_y_idx = recent_y_positions.index(min(recent_y_positions))
            if 1 <= min_y_idx <= 3:  # Dip occurred in middle of sequence
                # Check if ball was near rim during dip
                dip_position = self.ball_pos[-(5-min_y_idx)][0]
                distance_to_rim = math.sqrt(
                    (dip_position[0] - rim_center[0]) ** 2 + 
                    (dip_position[1] - rim_center[1]) ** 2
                )
                if distance_to_rim <= rim_width * 1.2:
                    return True
        
        return False

    def add_vlm_annotations(self):
        """
        Add comprehensive annotations for VLM analysis
        """
        frame_height, frame_width = self.frame.shape[:2]
        
        # Add frame information box in top-left corner
        info_box_height = 120
        cv2.rectangle(self.frame, (0, 0), (300, info_box_height), (0, 0, 0), -1)  # Black background
        cv2.rectangle(self.frame, (0, 0), (300, info_box_height), (255, 255, 255), 2)  # White border
        
        # Frame and time info
        time_seconds = self.frame_count / self.video_fps
        minutes = int(time_seconds // 60)
        seconds = int(time_seconds % 60)
        
        info_lines = [
            f"Frame: {self.frame_count}",
            f"Time: {minutes:02d}:{seconds:02d}",
            f"Balls: {len(self.ball_pos)}",
            f"Rims: {len(self.hoop_pos)}",
            f"Attempts: {self.attempts}",
            f"Makes: {self.makes}"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(self.frame, line, (10, 20 + i * 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add detection state box in top-right corner
        state_box_width = 250
        cv2.rectangle(self.frame, (frame_width - state_box_width, 0), 
                     (frame_width, info_box_height), (0, 0, 0), -1)  # Black background
        cv2.rectangle(self.frame, (frame_width - state_box_width, 0), 
                     (frame_width, info_box_height), (255, 255, 255), 2)  # White border
        
        state_lines = [
            f"Shot State: {'IN PROGRESS' if self.shot_in_progress else 'IDLE'}",
            f"Up Detected: {self.up}",
            f"Down Detected: {self.down}",
            f"Up Frames: {self.frames_since_up}",
            f"Down Frames: {self.frames_since_down}",
            f"Near Rim: {self.ball_near_rim_count}"
        ]
        
        for i, line in enumerate(state_lines):
            cv2.putText(self.frame, line, (frame_width - state_box_width + 10, 20 + i * 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add ball trajectory annotations
        if len(self.ball_pos) > 1:
            # Draw trajectory lines between ball positions
            for i in range(1, len(self.ball_pos)):
                cv2.line(self.frame, self.ball_pos[i-1][0], self.ball_pos[i][0], (0, 255, 255), 2)
            
            # Annotate ball velocity and direction
            if len(self.ball_pos) >= 2:
                curr_pos = self.ball_pos[-1][0]
                prev_pos = self.ball_pos[-2][0]
                dx = curr_pos[0] - prev_pos[0]
                dy = curr_pos[1] - prev_pos[1]
                velocity = math.sqrt(dx*dx + dy*dy)
                
                # Add velocity annotation near ball
                cv2.putText(self.frame, f"V:{velocity:.1f}", 
                           (curr_pos[0] + 20, curr_pos[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Add rim area annotations
        if len(self.hoop_pos) > 0:
            rim_center = self.hoop_pos[-1][0]
            rim_width = self.hoop_pos[-1][2]
            rim_height = self.hoop_pos[-1][3]
            
            # Draw scoring zone
            score_zone_w = int(rim_width * DetectionConfig.SCORE_ZONE_MULTIPLIER)
            score_zone_h = int(rim_height * DetectionConfig.SCORE_ZONE_MULTIPLIER)
            cv2.rectangle(self.frame, 
                         (rim_center[0] - score_zone_w//2, rim_center[1] - score_zone_h//2),
                         (rim_center[0] + score_zone_w//2, rim_center[1] + score_zone_h//2),
                         (0, 255, 0), 2)
            cv2.putText(self.frame, "SCORE ZONE", 
                       (rim_center[0] - 40, rim_center[1] - score_zone_h//2 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw up detection zone
            up_zone_w = int(rim_width * DetectionConfig.UP_ZONE_MULTIPLIER)
            up_zone_h = int(rim_height * DetectionConfig.UP_ZONE_MULTIPLIER)
            cv2.rectangle(self.frame, 
                         (rim_center[0] - up_zone_w//2, rim_center[1] - up_zone_h//2),
                         (rim_center[0] + up_zone_w//2, rim_center[1] + up_zone_h//2),
                         (255, 0, 0), 1)
            cv2.putText(self.frame, "UP DETECT ZONE", 
                       (rim_center[0] - 50, rim_center[1] + up_zone_h//2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Add event annotations for recent shots
        if self.fade_counter > 0:
            # Large, prominent shot result annotation
            result_text = self.overlay_text
            text_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 6)[0]
            text_x = (frame_width - text_size[0]) // 2
            text_y = frame_height // 2
            
            # Add background for better visibility
            cv2.rectangle(self.frame, 
                         (text_x - 20, text_y - 50), 
                         (text_x + text_size[0] + 20, text_y + 20),
                         (0, 0, 0), -1)
            cv2.putText(self.frame, result_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 3, self.overlay_color, 6)
            
            # Add frame number of the shot for VLM reference
            shot_info = f"Shot at Frame {self.last_detection_frame}"
            cv2.putText(self.frame, shot_info, (text_x, text_y + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    def display_score(self):
        # Add main score text with enhanced styling for low-angle view
        text = f"{self.makes} / {self.attempts}"
        
        # Add background rectangle for better text visibility
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 6)
        cv2.rectangle(self.frame, (40, 90), (40 + text_width + 20, 90 + text_height + 20), 
                     (0, 0, 0), -1)  # Black background
        cv2.rectangle(self.frame, (40, 90), (40 + text_width + 20, 90 + text_height + 20), 
                     (255, 255, 255), 2)  # White border
        
        # Main score text
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 6)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 100, 255), 3)

        # Add shooting percentage if attempts > 0
        if self.attempts > 0:
            percentage = (self.makes / self.attempts) * 100
            perc_text = f"{percentage:.1f}%"
            cv2.putText(self.frame, perc_text, (50, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(self.frame, perc_text, (50, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 255, 100), 2)

        # Add overlay text for shot result
        if hasattr(self, 'overlay_text') and self.overlay_text != "Waiting...":
            # Calculate text size to position it at the center top
            (text_width, text_height), _ = cv2.getTextSize(self.overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 6)
            text_x = (self.frame.shape[1] - text_width) // 2  # Center alignment
            text_y = 80  # Top margin

            # Add background for overlay text
            cv2.rectangle(self.frame, 
                         (text_x - 20, text_y - text_height - 10), 
                         (text_x + text_width + 20, text_y + 10),
                         (0, 0, 0), -1)

            # Display overlay text with color
            cv2.putText(self.frame, self.overlay_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2.5,
                        self.overlay_color, 6)

        # Enhanced debug visualization
        if self.debug_mode:
            debug_y_start = self.frame.shape[0] - 120
            
            # Shot state information
            status_lines = [
                f"Balls: {len(self.ball_pos)}, Hoops: {len(self.hoop_pos)}",
                f"Up: {self.up} (frames: {self.frames_since_up})",
                f"Down: {self.down} (frames: {self.frames_since_down})",
                f"Near rim: {self.ball_near_rim_count} (max: {self.max_ball_near_rim})"
            ]
            
            for i, line in enumerate(status_lines):
                cv2.putText(self.frame, line, (10, debug_y_start + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Ball confidence info
            if len(self.ball_pos) > 0:
                last_ball_conf = self.ball_pos[-1][4]
                cv2.putText(self.frame, f"Ball conf: {last_ball_conf:.2f}", 
                           (10, debug_y_start - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
        
        # Main status display
        status_text = ""
        if self.shot_in_progress:
            if self.up and not self.down:
                status_text = "TRACKING SHOT"
            elif self.up and self.down:
                status_text = "ANALYZING"
        elif len(self.ball_pos) > 0 and len(self.hoop_pos) > 0:
            status_text = "READY"
        else:
            status_text = "SEARCHING..."
        
        cv2.putText(self.frame, status_text, (50, self.frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Gradually fade out color after shot
        if self.fade_counter > 0:
            alpha = 0.15 * (self.fade_counter / self.fade_frames)  # Reduced intensity
            overlay = np.full_like(self.frame, self.overlay_color)
            self.frame = cv2.addWeighted(self.frame, 1 - alpha, overlay, alpha, 0)
            self.fade_counter -= 1


if __name__ == "__main__":
    # You can specify different model and video paths
    detector = ShotDetectorLowAngle(
        model_path="300images-yolom11/best.pt",  # Update this path to your trained model
        video_path="sample180s_video-1.mp4",    # Update this path to your test video
        save_output=True                         # Set to True to save analyzed video
    )
