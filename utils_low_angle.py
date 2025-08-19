import math
import numpy as np
import torch

def get_device():
    """Automatically select devices -> mps（Mac） -> cpu"""
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device


def score(ball_pos, hoop_pos):
    """
    Enhanced scoring logic for low-angle court view with multiple detection methods.
    """
    if len(ball_pos) < 3 or len(hoop_pos) < 1:
        return False
    
    # Get rim center and dimensions
    rim_center_x = hoop_pos[-1][0][0]
    rim_center_y = hoop_pos[-1][0][1]
    rim_width = hoop_pos[-1][2]
    rim_height = hoop_pos[-1][3]
    
    # Method 1: Direct position check with larger scoring zone
    score_zone_x = rim_width * 1.0  # Increased from 0.8 to 1.0
    score_zone_y = rim_height * 1.0  # Increased from 0.8 to 1.0
    
    # Check recent ball positions for rim passage
    for i in range(min(15, len(ball_pos))):  # Check last 15 positions (increased)
        ball_x = ball_pos[-(i+1)][0][0]
        ball_y = ball_pos[-(i+1)][0][1]
        
        # Check if ball passed through scoring zone
        if (abs(ball_x - rim_center_x) <= score_zone_x and 
            abs(ball_y - rim_center_y) <= score_zone_y):
            return True
    
    # Method 2: Trajectory intersection analysis
    if len(ball_pos) >= 4:
        # Check multiple trajectory segments
        for i in range(min(8, len(ball_pos) - 1)):  # Check more segments
            x1, y1 = ball_pos[-(i+2)][0]
            x2, y2 = ball_pos[-(i+1)][0]
            
            # Check if line segment passes near rim
            dist_to_rim = point_to_line_distance(rim_center_x, rim_center_y, x1, y1, x2, y2)
            if dist_to_rim <= max(score_zone_x, score_zone_y) * 1.2:  # 20% more generous
                return True
    
    # Method 3: Check for ball "disappearance" near rim (occlusion by rim)
    if len(ball_pos) >= 5:
        # Look for gaps in ball detection near rim
        rim_area_visits = []
        for i, ball_data in enumerate(ball_pos[-10:]):  # Last 10 positions
            ball_x, ball_y = ball_data[0]
            distance_to_rim = ((ball_x - rim_center_x) ** 2 + (ball_y - rim_center_y) ** 2) ** 0.5
            if distance_to_rim <= max(rim_width, rim_height) * 1.5:
                rim_area_visits.append(i)
        
        # If ball was detected near rim multiple times, likely a make
        if len(rim_area_visits) >= 3:
            return True
    
    return False


def point_to_line_distance(px, py, x1, y1, x2, y2):
    """Calculate distance from point to line segment"""
    A = px - x1
    B = py - y1
    C = x2 - x1
    D = y2 - y1
    
    dot = A * C + B * D
    len_sq = C * C + D * D
    
    if len_sq == 0:
        return math.sqrt(A * A + B * B)
    
    param = dot / len_sq
    
    if param < 0:
        xx, yy = x1, y1
    elif param > 1:
        xx, yy = x2, y2
    else:
        xx, yy = x1 + param * C, y1 + param * D
    
    dx = px - xx
    dy = py - yy
    return math.sqrt(dx * dx + dy * dy)


def detect_down(ball_pos, hoop_pos):
    """
    Adapted for low-angle view: detect when ball is moving away from rim
    or has passed the rim area (indicating end of shot attempt)
    """
    if len(ball_pos) < 2 or len(hoop_pos) < 1:
        return False
    
    rim_center_x = hoop_pos[-1][0][0]
    rim_center_y = hoop_pos[-1][0][1]
    rim_width = hoop_pos[-1][2]
    
    current_ball = ball_pos[-1][0]
    
    # Define "down" zone - area past the rim indicating shot completion
    # This could be either side of the rim depending on shot direction
    down_zone_distance = rim_width * 3  # 3x rim width as buffer
    
    # Check if ball is far enough from rim center (horizontally or vertically)
    distance_from_rim = math.sqrt(
        (current_ball[0] - rim_center_x) ** 2 + 
        (current_ball[1] - rim_center_y) ** 2
    )
    
    return distance_from_rim > down_zone_distance


def detect_up(ball_pos, hoop_pos):
    """
    Enhanced detection for ball approaching rim area in low-angle view
    """
    if len(ball_pos) < 1 or len(hoop_pos) < 1:
        return False
    
    rim_center_x = hoop_pos[-1][0][0]
    rim_center_y = hoop_pos[-1][0][1]
    rim_width = hoop_pos[-1][2]
    rim_height = hoop_pos[-1][3]
    
    current_ball = ball_pos[-1][0]
    
    # Larger detection zone for better sensitivity
    up_zone_x = rim_width * 5   # Increased from 4x to 5x
    up_zone_y = rim_height * 5  # Increased from 4x to 5x
    
    # Check if ball is in the approach zone
    in_approach_zone = (
        abs(current_ball[0] - rim_center_x) <= up_zone_x and
        abs(current_ball[1] - rim_center_y) <= up_zone_y
    )
    
    if not in_approach_zone:
        return False
    
    # If we only have one ball position, return true if in zone
    if len(ball_pos) < 2:
        return True
    
    # Enhanced movement analysis
    if len(ball_pos) >= 3:
        # Check movement over last 3 positions for better accuracy
        distances = []
        for i in range(min(3, len(ball_pos))):
            ball_x, ball_y = ball_pos[-(i+1)][0]
            distance = math.sqrt(
                (ball_x - rim_center_x) ** 2 + 
                (ball_y - rim_center_y) ** 2
            )
            distances.append(distance)
        
        # Check if generally moving toward rim (allow for some noise)
        moving_toward = sum(distances[i] > distances[i+1] for i in range(len(distances)-1))
        if moving_toward >= len(distances) // 2:  # At least half the movements toward rim
            return True
    
    # Fallback: simpler check with just last two positions
    if len(ball_pos) >= 2:
        prev_ball = ball_pos[-2][0]
        prev_distance = math.sqrt(
            (prev_ball[0] - rim_center_x) ** 2 + 
            (prev_ball[1] - rim_center_y) ** 2
        )
        current_distance = math.sqrt(
            (current_ball[0] - rim_center_x) ** 2 + 
            (current_ball[1] - rim_center_y) ** 2
        )
        
        return current_distance <= prev_distance  # Allow equal distance
    
    return True  # Default to true if in zone


def in_hoop_region(center, hoop_pos):
    """
    Adapted for low-angle view: more generous hoop region due to smaller rim appearance
    """
    if len(hoop_pos) < 1:
        return False
    
    x = center[0]
    y = center[1]
    
    rim_center_x = hoop_pos[-1][0][0]
    rim_center_y = hoop_pos[-1][0][1]
    rim_width = hoop_pos[-1][2]
    rim_height = hoop_pos[-1][3]
    
    # More generous region - 2x the rim dimensions
    region_width = rim_width * 2
    region_height = rim_height * 2
    
    x1 = rim_center_x - region_width / 2
    x2 = rim_center_x + region_width / 2
    y1 = rim_center_y - region_height / 2
    y2 = rim_center_y + region_height / 2
    
    return x1 < x < x2 and y1 < y < y2


def clean_ball_pos(ball_pos, frame_count):
    """
    Enhanced ball position cleaning for low-angle view with stricter size validation
    """
    if len(ball_pos) > 1:
        # Width and Height
        w1 = ball_pos[-2][2]
        h1 = ball_pos[-2][3]
        w2 = ball_pos[-1][2]
        h2 = ball_pos[-1][3]

        # X and Y coordinates
        x1 = ball_pos[-2][0][0]
        y1 = ball_pos[-2][0][1]
        x2 = ball_pos[-1][0][0]
        y2 = ball_pos[-1][0][1]

        # Frame count
        f1 = ball_pos[-2][1]
        f2 = ball_pos[-1][1]
        f_dif = f2 - f1

        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Adjusted for low-angle view - ball movement should be more constrained
        max_dist = 3 * math.sqrt((w1) ** 2 + (h1) ** 2)

        # Ball should not move 3x its diameter within 5 frames (reduced from 4x)
        if (dist > max_dist) and (f_dif < 5):
            ball_pos.pop()

        # Stricter aspect ratio check for low-angle view
        elif (w2 * 1.2 < h2) or (h2 * 1.2 < w2):  # Changed from 1.4 to 1.2
            ball_pos.pop()

    # Keep tracking history slightly longer for better trajectory analysis
    if len(ball_pos) > 0:
        if frame_count - ball_pos[0][1] > 40:  # Increased from 30 to 40
            ball_pos.pop(0)

    return ball_pos


def clean_hoop_pos(hoop_pos):
    """
    Enhanced hoop position cleaning for low-angle view
    """
    if len(hoop_pos) > 1:
        x1 = hoop_pos[-2][0][0]
        y1 = hoop_pos[-2][0][1]
        x2 = hoop_pos[-1][0][0]
        y2 = hoop_pos[-1][0][1]

        w1 = hoop_pos[-2][2]
        h1 = hoop_pos[-2][3]
        w2 = hoop_pos[-1][2]
        h2 = hoop_pos[-1][3]

        f1 = hoop_pos[-2][1]
        f2 = hoop_pos[-1][1]
        f_dif = f2 - f1

        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)

        # Stricter movement constraint for low-angle rim detection
        max_dist = 0.3 * math.sqrt(w1 ** 2 + h1 ** 2)  # Reduced from 0.5

        # Hoop should not move 0.3x its diameter within 5 frames
        if dist > max_dist and f_dif < 5:
            hoop_pos.pop()

        # Stricter aspect ratio for rim (should be more circular from this angle)
        if (w2 * 1.1 < h2) or (h2 * 1.1 < w2):  # Changed from 1.3 to 1.1
            hoop_pos.pop()

    # Keep more hoop positions for stable tracking
    if len(hoop_pos) > 35:  # Increased from 25
        hoop_pos.pop(0)

    return hoop_pos
