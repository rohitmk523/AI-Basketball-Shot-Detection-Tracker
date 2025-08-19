#!/usr/bin/env python3
"""
Fast processing script for generating VLM-ready analyzed videos
Disables display window for faster processing
"""

from shot_detector_low_angle import ShotDetectorLowAngle
from detection_config import DetectionConfig, ConfigPresets
import cv2

class FastShotDetector(ShotDetectorLowAngle):
    """Modified detector that doesn't show display window for faster processing"""
    
    def run(self):
        print("Processing video for VLM analysis...")
        print("Display window disabled for faster processing")
        
        frame_count = 0
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, self.frame = self.cap.read()

            if not ret:
                break

            # Show progress every 100 frames
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"Progress: {progress:.1f}% (Frame {frame_count}/{total_frames})")

            # Process frame (same as parent class)
            results = self.model(self.frame, stream=True, device=self.device)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    # Confidence
                    import math
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # Class Name
                    cls = int(box.cls[0])
                    current_class = self.class_names[cls]

                    center = (int(x1 + w / 2), int(y1 + h / 2))

                    # Use configurable confidence thresholds
                    ball_threshold = DetectionConfig.BALL_CONFIDENCE_THRESHOLD
                    ball_near_hoop_threshold = DetectionConfig.BALL_NEAR_HOOP_CONFIDENCE
                    
                    if (conf > ball_threshold or (self.in_hoop_region_local(center) and conf > ball_near_hoop_threshold)) and current_class == "Basketball":
                        self.ball_pos.append((center, self.frame_count, w, h, conf))
                        # Add detection box
                        import cvzone
                        cvzone.cornerRect(self.frame, (x1, y1, w, h), colorR=(255, 100, 0))
                        
                        # Track ball near rim
                        if len(self.hoop_pos) > 0 and self.in_hoop_region_local(center):
                            self.ball_near_rim_count += 1
                            self.max_ball_near_rim = max(self.max_ball_near_rim, self.ball_near_rim_count)
                        else:
                            self.ball_near_rim_count = 0
                        
                        # Debug info
                        if self.debug_mode:
                            cv2.putText(self.frame, f"Ball: {conf:.2f}", (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)

                    # Rim detection
                    if conf > DetectionConfig.HOOP_CONFIDENCE_THRESHOLD and current_class == "Basketball Hoop":
                        self.hoop_pos.append((center, self.frame_count, w, h, conf))
                        cvzone.cornerRect(self.frame, (x1, y1, w, h), colorR=(0, 255, 100))
                        
                        if self.debug_mode:
                            cv2.putText(self.frame, f"Rim: {conf:.2f}", (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 100), 1)

            self.clean_motion()
            self.shot_detection()
            self.add_vlm_annotations()
            self.display_score()
            self.frame_count += 1
            frame_count += 1

            # Save frame to output video
            if self.save_output and self.video_writer is not None:
                self.video_writer.write(self.frame)

            # No display window - just process and save

        # Cleanup
        self.cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"\n✅ Analysis complete!")
            print(f"📹 Analyzed video saved as: {self.output_path}")
            print(f"📊 Final stats: {self.makes}/{self.attempts} shots ({(self.makes/max(self.attempts,1)*100):.1f}%)")
            print(f"🎯 Ready for VLM analysis with Gemini 2.5 Pro!")
            print(f"💡 See VLM_ANALYSIS_PROMPT.md for analysis instructions")
    
    def in_hoop_region_local(self, center):
        """Local version of in_hoop_region to avoid import issues"""
        if len(self.hoop_pos) < 1:
            return False
        
        x, y = center
        rim_center_x = self.hoop_pos[-1][0][0]
        rim_center_y = self.hoop_pos[-1][0][1]
        rim_width = self.hoop_pos[-1][2]
        rim_height = self.hoop_pos[-1][3]
        
        region_width = rim_width * 2
        region_height = rim_height * 2
        
        x1 = rim_center_x - region_width / 2
        x2 = rim_center_x + region_width / 2
        y1 = rim_center_y - region_height / 2
        y2 = rim_center_y + region_height / 2
        
        return x1 < x < x2 and y1 < y < y2


def main():
    """Main function to process video for VLM analysis"""
    
    print("🏀 Basketball Shot Detection - VLM Processing Mode")
    print("=" * 50)
    
    # Configuration
    model_path = "300images-yolom11/best.pt"
    video_path = "sample180s_video-1.mp4"
    
    # Apply high sensitivity for better detection
    ConfigPresets.high_sensitivity()
    
    # Disable debug mode for cleaner output video
    DetectionConfig.DEBUG_MODE = False
    
    print(f"📁 Input video: {video_path}")
    print(f"🤖 Model: {model_path}")
    print("⚙️  Using high sensitivity configuration")
    print("🚀 Starting fast processing...")
    
    # Process video
    detector = FastShotDetector(
        model_path=model_path,
        video_path=video_path,
        save_output=True
    )

if __name__ == "__main__":
    main()
