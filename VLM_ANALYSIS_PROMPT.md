# VLM Analysis Prompt for Basketball Shot Detection

## Prompt Template for Gemini 2.5 Pro

Use this prompt when analyzing the generated video with Gemini 2.5 Pro:

```
I have a basketball video with computer vision annotations that I need you to analyze for shot detection and scoring. The video contains the following visual elements:

**Visual Annotations Explained:**
- Orange bounding boxes: Detected basketballs with confidence scores
- Green bounding boxes: Detected basketball rims with confidence scores
- Yellow trajectory lines: Ball movement path
- Green rectangles around rim: Scoring zones where successful shots are detected
- Red rectangles around rim: Shot approach detection zones
- Top-left info box: Frame number, timestamp, detection counts, attempts, and makes
- Top-right state box: Current detection state (shot in progress, up/down detection status)
- Large center text: Shot results (MAKE/MISS) when detected
- Blue circles: Ball position history
- Yellow circle: Current rim center

**What I need you to analyze:**
1. **Shot Identification**: Identify all shooting attempts in the video, even if the computer vision system missed them
2. **Shot Outcomes**: For each shot, determine if it was made or missed
3. **Shot Timing**: Provide frame numbers or timestamps for when shots occur
4. **Accuracy Assessment**: Compare your analysis with the computer vision system's detected attempts and makes
5. **False Positives/Negatives**: Identify any incorrect detections by the CV system

**Specific Questions:**
1. How many total shooting attempts do you observe in the video?
2. How many of these attempts resulted in successful shots (makes)?
3. What is the actual shooting percentage?
4. Are there any shots that the computer vision system missed detecting?
5. Are there any false positive detections (system thought there was a shot when there wasn't)?
6. At what specific frames/timestamps do the actual shots occur?
7. For missed shots by the CV system, what might have caused the detection failure?

**Please provide:**
- A chronological list of all shots with frame numbers and outcomes
- Comparison between CV system results and your visual analysis
- Recommendations for improving the detection system
- Any patterns you notice in missed detections

**Frame Reference:**
- The video runs at [FPS] frames per second
- Frame numbers and timestamps are shown in the top-left corner
- Use these for precise shot timing references

Please analyze the entire video carefully and provide a comprehensive shot-by-shot breakdown.
```

## Usage Instructions

1. **Generate the analyzed video** by running:
   ```python
   detector = ShotDetectorLowAngle(
       model_path="300images-yolom11/best.pt",
       video_path="sample180s_video-1.mp4",
       save_output=True
   )
   ```

2. **Upload the generated video** (e.g., `sample180s_video-1_analyzed.mp4`) to Gemini 2.5 Pro

3. **Use the prompt above** to get detailed analysis

4. **Follow-up questions** you can ask:
   - "Can you create a shot chart showing where each attempt was taken from?"
   - "What specific visual cues indicate a successful shot versus a miss?"
   - "At what frame ranges should I look to debug the detection system?"
   - "Are there any patterns in ball movement that indicate shot attempts?"

## Expected VLM Output

The VLM should provide:
- **Shot Log**: Frame-by-frame breakdown of all shooting attempts
- **Accuracy Comparison**: CV system vs. actual performance
- **Miss Analysis**: Why certain shots weren't detected
- **Improvement Suggestions**: How to enhance the detection algorithm

## Post-Analysis Actions

Based on VLM feedback, you can:
1. **Adjust detection parameters** in `detection_config.py`
2. **Modify zone sizes** for better shot detection
3. **Fine-tune confidence thresholds** based on missed detections
4. **Update timing parameters** for better shot sequence detection

This approach leverages the VLM's superior visual understanding to validate and improve your computer vision system's performance.
