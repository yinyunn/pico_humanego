import cv2
import numpy as np
from typing import List, Tuple

class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    HEADER = f"{BOLD}{BLUE}"
    OKBLUE = f"{BOLD}{CYAN}"
    OKGREEN = f"{BOLD}{GREEN}"
    WARNING = f"{BOLD}{YELLOW}"
    FAIL = f"{BOLD}{RED}"
    ENDC = END
    
    
C_CYAN = (255, 255, 0); C_GREEN = (0, 255, 0); C_RED = (0, 0, 255)
C_GOLD = (0, 215, 255); C_WHITE = (255, 255, 255); C_GRAY = (100, 100, 100)


def draw_glass_rect(img, pt1, pt2, alpha=0.6):
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
    cv2.rectangle(img, pt1, pt2, (180, 180, 180), 1, cv2.LINE_AA)
    return img


def draw_arc_gauge(img, center, value, vmax, label):
    cx, cy = center
    r = 22
    v = float(np.clip(value / (vmax + 1e-6), 0.0, 1.0))
    cv2.ellipse(img, (cx, cy), (r, r), 0, 225, -45, (60, 60, 60), 3, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy), (r, r), 0, 225, 225 + int(270 * v), (0, 200, 255), 3, cv2.LINE_AA)
    cv2.putText(img, label, (cx - 6, cy + 6), cv2.FONT_HERSHEY_DUPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)


def draw_status_bar(img, pos, width, val, max_val, label, color):
    x, y = pos
    bar_w = int((val / max_val) * width) if max_val > 0 else 0
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_WHITE, 1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + width, y + 5), (50, 50, 50), -1)
    cv2.rectangle(img, (x, y), (x + bar_w, y + 5), color, -1)






def draw_text_with_shadow(img: np.ndarray, text: str, position: Tuple[int, int],
                          font_scale: float, color: Tuple[int, int, int],
                          thickness: int = 2) -> None:
    """
    Draws text with a dark shadow background for better readability on complex camera streams.
    
    Args:
        img (np.ndarray): The OpenCV image canvas.
        text (str): Text to display.
        position (Tuple[int, int]): Bottom-left corner of the text string (x, y).
        font_scale (float): Font scale factor.
        color (Tuple[int, int, int]): Main text color in BGR format.
        thickness (int): Thickness of the main text.
    """
    font = cv2.FONT_HERSHEY_DUPLEX
    x, y = position
    
    # Draw shadows (offset by +1 and -1)
    cv2.putText(img, text, (x + 1, y + 1), font, font_scale, (20, 20, 20), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x - 1, y - 1), font, font_scale, (20, 20, 20), thickness + 2, cv2.LINE_AA)
    
    # Draw actual foreground text
    cv2.putText(img, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

def get_traj_colors(horizon: int) -> List[Tuple[int, int, int]]:
    """
    Generates a color gradient list for trajectory visualization.
    Fades gracefully to represent time steps into the future.
    
    Args:
        horizon (int): Number of steps in the predicted trajectory.
        
    Returns:
        List[Tuple[int, int, int]]: A list of BGR colors.
    """
    return[
        (int(bg), int(bg), int(r)) 
        for bg, r in zip(np.linspace(200, 0, horizon), np.linspace(255, 100, horizon))
    ]

def project_3d_to_2d(p_3d: np.ndarray, K_mat: np.ndarray) -> Tuple[int, int]:
    """
    Projects a 3D point in the camera coordinate system to a 2D pixel coordinate.
    
    Args:
        p_3d (np.ndarray): 3D point [X, Y, Z].
        K_mat (np.ndarray): 3x3 Camera intrinsic matrix.
        
    Returns:
        Tuple[int, int]: (u, v) pixel coordinates.
    """
    if p_3d[2] < 1e-6:  # Prevent division by zero
        return (0, 0)
    
    u_proj = int(K_mat[0, 0] * p_3d[0] / p_3d[2] + K_mat[0, 2])
    v_proj = int(K_mat[1, 1] * p_3d[1] / p_3d[2] + K_mat[1, 2])
    
    return (u_proj, v_proj)