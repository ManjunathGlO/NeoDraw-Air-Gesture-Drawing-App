import cv2
import numpy as np
import mediapipe as mp
import time
import os

# ------------------ Settings ------------------
WIDTH, HEIGHT = 1280, 720
FPS = 30

# Colors for palette (BGR)
PALETTE = [
    (255, 255, 255),  # White
    (0, 0, 255),      # Red
    (0, 165, 255),    # Orange
    (0, 255, 255),    # Yellow
    (0, 255, 0),      # Green
    (255, 0, 255),    # Magenta
    (255, 0, 0),      # Blue
    (0, 0, 0),        # Black (useful)
]

DEFAULT_COLOR = (0, 0, 255)  # Starting neon color (red)
MIN_BRUSH, MAX_BRUSH = 4, 80  # pixel range for brush size

# ------------------ Mediapipe Setup ------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

# ------------------ Canvas Layers ------------------
canvas_base = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)  # actual strokes
glow_layer = np.zeros_like(canvas_base)                     # blurred glow
composite = np.zeros_like(canvas_base)                      # merged output for display

# ------------------ State ------------------
# ------------------ State ------------------
draw_color = DEFAULT_COLOR
eraser_color = (0, 0, 0)
mode = "Idle"
prev_x, prev_y = None, None
smoothing = 6  # higher = smoother
last_save_time = 0
brush_size = 20   # ⭐ important: default brush size to avoid NameError


# ------------------ Helpers ------------------
def finger_states(landmarks):
    # Return list of 5 booleans for Thumb->Pinky (True if extended)
    # Use simple comparison: tip_y < pip_y for index/middle/ring/pinky
    # For thumb, use tip_x relative to ip_x (handedness not considered, but works for mirror-frame)
    tips = [mp_hands.HandLandmark.THUMB_TIP,
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP]
    pips = [mp_hands.HandLandmark.THUMB_IP,
            mp_hands.HandLandmark.INDEX_FINGER_PIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            mp_hands.HandLandmark.RING_FINGER_PIP,
            mp_hands.HandLandmark.PINKY_PIP]

    states = []
    for tip, pip in zip(tips, pips):
        tip_lm = landmarks.landmark[tip]
        pip_lm = landmarks.landmark[pip]
        # Thumb logic: use x (because thumb extends sideways)
        if tip == mp_hands.HandLandmark.THUMB_TIP:
            states.append(tip_lm.x < pip_lm.x)  # mirror frame: smaller x is extended to the right hand
        else:
            states.append(tip_lm.y < pip_lm.y)
    return states  # [thumb, index, middle, ring, pinky]

def map_range(val, in_min, in_max, out_min, out_max):
    val = max(in_min, min(in_max, val))
    return int(out_min + (out_max - out_min) * ((val - in_min) / (in_max - in_min)))

def draw_neon_line(layer_base, glow, p1, p2, color, thickness):
    # Draw core line (sharp)
    cv2.line(layer_base, p1, p2, color, max(1, thickness // 2), cv2.LINE_AA)
    # Draw thicker translucent line on glow layer
    glow_color = color
    cv2.line(glow, p1, p2, glow_color, thickness, cv2.LINE_AA)

def save_canvas(img, folder="captures"):
    os.makedirs(folder, exist_ok=True)
    t = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(folder, f"neon_drawing_{t}.png")
    cv2.imwrite(path, img)
    return path

# ------------------ Video Capture ------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Palette geometry
palette_y0 = 80
palette_h = 60
palette_w = 80
palette_x0 = 20
palette_spacing = 12

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        h, w, _ = frame.shape

        # Draw UI: Top bar background
        ui = frame.copy()
        cv2.rectangle(ui, (0, 0), (WIDTH, 80), (30, 30, 30), -1)  # top bar

        # Draw palette
        x = palette_x0
        for i, col in enumerate(PALETTE):
            rect_tl = (x, palette_y0)
            rect_br = (x + palette_w, palette_y0 + palette_h)
            cv2.rectangle(ui, rect_tl, rect_br, (50,50,50), -1)
            # inner color box
            cv2.rectangle(ui, (x+6, palette_y0+6), (x+palette_w-6, palette_y0+palette_h-6), col, -1)
            # border if selected
            if col == draw_color:
                cv2.rectangle(ui, (x, palette_y0), (x + palette_w, palette_y0 + palette_h), (255,255,255), 3)
            x += palette_w + palette_spacing

        # Toolbar text
        cv2.putText(ui, f"Mode: {mode}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230,230,230), 2, cv2.LINE_AA)
        cv2.putText(ui, "1 finger: Draw  |  2 fingers: Pick Color  |  Pinch: Erase  |  S: Save  C: Clear", (260, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1, cv2.LINE_AA)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(ui, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(170,220,255), thickness=2, circle_radius=3),
                                          mp_drawing.DrawingSpec(color=(255,120,255), thickness=2))

                # Get index & thumb coords
                idx_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                mid_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                x_index, y_index = int(idx_tip.x * w), int(idx_tip.y * h)
                x_mid, y_mid = int(mid_tip.x * w), int(mid_tip.y * h)
                x_thumb, y_thumb = int(thumb_tip.x * w), int(thumb_tip.y * h)

                # Smooth coordinates
                if prev_x is not None:
                    x_index = prev_x + (x_index - prev_x) // smoothing
                    y_index = prev_y + (y_index - prev_y) // smoothing

                # Determine finger extension states
                fstates = finger_states(hand_landmarks)
                fingers_up = sum(fstates[1:])  # count index..pinky extended (ignore thumb)
                index_up = fstates[1]
                middle_up = fstates[2]

                # Distance thumb-index -> pinch detection
                pinch_dist = int(np.hypot(x_thumb - x_index, y_thumb - y_index))

                # Brush size control: use distance between index and middle fingers (larger -> bigger brush)
                inter_dist = int(np.hypot(x_index - x_mid, y_index - y_mid))
                brush_size = map_range(inter_dist, 20, 220, MIN_BRUSH, MAX_BRUSH)

                # Mode logic
                if pinch_dist < 40:
                    mode = "Erase"
                    # draw erase on both base and glow by drawing black thick stroke
                    if prev_x is not None:
                        draw_neon_line(canvas_base, glow_layer, (prev_x, prev_y), (x_index, y_index), eraser_color, max(brush_size, 40))
                elif index_up and not middle_up:
                    mode = "Draw"
                    if prev_x is not None:
                        draw_neon_line(canvas_base, glow_layer, (prev_x, prev_y), (x_index, y_index), draw_color, brush_size)
                elif index_up and middle_up:
                    # Two fingers extended => color pick mode when hovering palette
                    mode = "Color"
                    # Check if index finger is on palette region
                    if palette_y0 <= y_index <= palette_y0 + palette_h:
                        # compute which palette rect
                        rel_x = x_index - palette_x0
                        idx = rel_x // (palette_w + palette_spacing)
                        if 0 <= idx < len(PALETTE):
                            # pick color on hold - require small dwell to avoid accidental change
                            draw_color = PALETTE[int(idx)]
                else:
                    mode = "Idle"

                prev_x, prev_y = x_index, y_index

                # Visual pointer circle
                if mode == "Color":
                    cv2.circle(ui, (x_index, y_index), 14, (255,255,255), 2, cv2.LINE_AA)
                    cv2.circle(ui, (x_index, y_index), 10, draw_color, -1)
                elif mode == "Erase":
                    cv2.circle(ui, (x_index, y_index), 18, (255,255,255), 2, cv2.LINE_AA)
                    cv2.circle(ui, (x_index, y_index), 14, eraser_color, -1)
                elif mode == "Draw":
                    cv2.circle(ui, (x_index, y_index), 12, (255,255,255), 2, cv2.LINE_AA)
                    cv2.circle(ui, (x_index, y_index), 8, draw_color, -1)

        else:
            prev_x, prev_y = None, None
            mode = "Idle"

        # -------------- Glow effect --------------
        # Blur the glow layer to create neon effect and composite
        blurred = cv2.GaussianBlur(glow_layer, (0,0), sigmaX=25, sigmaY=25)
        # increase intensity of blurred glow
        glow_strength = 0.9
        composite = cv2.addWeighted(canvas_base, 1.0, blurred, glow_strength, 0)
        # merge with UI frame (camera) to produce final output
        output = cv2.addWeighted(ui, 0.45, composite, 0.55, 0)

        # small overlay note about brush size
        cv2.putText(output, f"Brush: {int(brush_size)} px", (20, HEIGHT-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240,240,240), 2, cv2.LINE_AA)

        # Show
        cv2.imshow("Neon Gesture Drawing", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            canvas_base = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            glow_layer = np.zeros_like(canvas_base)
        elif key == ord('s'):
            # Save composite (merge camera background with drawing) for a nicer screenshot
            merge_for_save = cv2.addWeighted(frame, 0.2, composite, 0.8, 0)
            path = save_canvas(merge_for_save)
            # quick flash feedback
            last_save_time = time.time()
            print("Saved:", path)
        elif key == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
