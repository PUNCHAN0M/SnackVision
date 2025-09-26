"""Pose keypoints visualization (selected joints only) using OpenCV.

Draw only:
 - Left Shoulder
 - Right Shoulder
 - Left Elbow
 - Right Elbow
 - Left Wrist
 - Right Wrist

Model: yolo11n-pose.pt (COCO 17 keypoints indexing assumed)
COCO keypoint indices (for reference):
0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle
"""

from ultralytics import YOLO
import cv2
from pathlib import Path

model = YOLO("./models/yolo11n-pose.pt")

video_path = Path("./videos/person.avi")
if not video_path.exists():
    raise FileNotFoundError(f"Video not found: {video_path}")
import time

# Selected keypoint indices we want to draw
SELECTED_KEYPOINTS = {
    5: "L_Shoulder",
    6: "R_Shoulder",
    7: "L_Elbow",
    8: "R_Elbow",
    9: "L_Wrist",
    10: "R_Wrist",
}

KEYPOINT_COLOR = (0, 255, 0)  # green
TEXT_COLOR = (0, 255, 255)  # yellow
POINT_RADIUS = 5
POINT_THICKNESS = -1
CONFIDENCE_THRESHOLD = 0.4
LINE_COLOR = (255, 0, 0)  # blue lines
LINE_THICKNESS = 2
BOUNDING_COLOR = (0, 0, 255)  # red bounding box
BOUNDING_THICKNESS = 2
BOUNDING_PADDING_RATIO = 0.10  # 10% padding
WRIST_PADDING_RATIO = 0.50  # 10% extra radius relative to elbow-wrist length
WRIST_OUTER_COLOR = (255, 255, 0)  # cyan/yellow outer ring
WRIST_OUTER_THICKNESS = 2


def draw_selected_keypoints(frame, keypoints_xy, keypoints_conf):
    frame_height, frame_width = frame.shape[:2]
    if keypoints_xy is None:
        return
    num_persons = keypoints_xy.shape[0]
    for person_index in range(num_persons):
        valid_points = []  # (kp_index, x, y)
        for kp_index, kp_label in SELECTED_KEYPOINTS.items():
            x, y = keypoints_xy[person_index, kp_index]
            conf = (
                keypoints_conf[person_index, kp_index]
                if keypoints_conf is not None
                else 1.0
            )
            if conf < CONFIDENCE_THRESHOLD or x <= 0 or y <= 0:
                continue
            valid_points.append((kp_index, float(x), float(y)))
            cv2.circle(
                frame,
                (int(x), int(y)),
                POINT_RADIUS,
                KEYPOINT_COLOR,
                POINT_THICKNESS,
            )
            cv2.putText(
                frame,
                SELECTED_KEYPOINTS[kp_index],
                (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )

        # ----------------- Connect arms (5-7-9) left and (6-8-10) right + shoulders 5-6 -----------------
        def safe_point(idx):
            for p in valid_points:
                if p[0] == idx:
                    return (int(p[1]), int(p[2]))
            return None

        left_chain = [5, 7, 9]
        right_chain = [6, 8, 10]
        connectors = []
        # shoulders connection
        connectors.append((5, 6))
        # arm segments
        connectors += list(zip(left_chain, left_chain[1:]))
        connectors += list(zip(right_chain, right_chain[1:]))
        for a_idx, b_idx in connectors:
            pa = safe_point(a_idx)
            pb = safe_point(b_idx)
            if pa is not None and pb is not None:
                cv2.line(frame, pa, pb, LINE_COLOR, LINE_THICKNESS, cv2.LINE_AA)

        # ----------------- Bounding box with 10% padding across arms (indices 5-10) -----------------
        # We consider only the indices we attempted (5..10). If enough points exist (>=2) build box.
        arm_points = [p for p in valid_points if 5 <= p[0] <= 10]
        if len(arm_points) >= 2:
            xs = [p[1] for p in arm_points]
            ys = [p[2] for p in arm_points]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            width = max_x - min_x
            height = max_y - min_y
            if width > 0 and height > 0:
                pad_x = width * BOUNDING_PADDING_RATIO
                pad_y = height * BOUNDING_PADDING_RATIO
                box_x1 = max(0, int(min_x - pad_x))
                box_y1 = max(0, int(min_y - pad_y))
                box_x2 = min(frame_width - 1, int(max_x + pad_x))
                box_y2 = min(frame_height - 1, int(max_y + pad_y))
                cv2.rectangle(
                    frame,
                    (box_x1, box_y1),
                    (box_x2, box_y2),
                    BOUNDING_COLOR,
                    BOUNDING_THICKNESS,
                )
                cv2.putText(
                    frame,
                    "Arms Region",
                    (box_x1, max(0, box_y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    BOUNDING_COLOR,
                    1,
                    cv2.LINE_AA,
                )

        # ----------------- Wrist padding circle (10% of elbow-wrist length) -----------------
        # left wrist (9) relative to left elbow (7)
        def find_point(idx):
            for p in valid_points:
                if p[0] == idx:
                    return p
            return None

        left_elbow = find_point(7)
        left_wrist = find_point(9)
        right_elbow = find_point(8)
        right_wrist = find_point(10)

        def draw_padded_wrist(elbow_point, wrist_point):
            if elbow_point is None or wrist_point is None:
                return
            _, ex, ey = elbow_point
            _, wx, wy = wrist_point
            distance_elbow_wrist = ((wx - ex) ** 2 + (wy - ey) ** 2) ** 0.5
            if distance_elbow_wrist <= 0:
                return
            extra_radius = int(distance_elbow_wrist * WRIST_PADDING_RATIO)
            base_radius = POINT_RADIUS + extra_radius
            cv2.circle(
                frame,
                (int(wx), int(wy)),
                base_radius,
                WRIST_OUTER_COLOR,
                WRIST_OUTER_THICKNESS,
                cv2.LINE_AA,
            )

        draw_padded_wrist(left_elbow, left_wrist)
        draw_padded_wrist(right_elbow, right_wrist)
    if keypoints_xy is None:
        return
    num_persons = keypoints_xy.shape[0]
    for person_index in range(num_persons):
        for kp_index, kp_label in SELECTED_KEYPOINTS.items():
            x, y = keypoints_xy[person_index, kp_index]
            conf = (
                keypoints_conf[person_index, kp_index]
                if keypoints_conf is not None
                else 1.0
            )
            if conf < CONFIDENCE_THRESHOLD:
                continue
            if x <= 0 or y <= 0:
                continue
            cv2.circle(
                frame, (int(x), int(y)), POINT_RADIUS, KEYPOINT_COLOR, POINT_THICKNESS
            )
            cv2.putText(
                frame,
                kp_label,
                (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )


def main():
    video_capture = cv2.VideoCapture(str(video_path))
    if not video_capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    window_name = "Selected Pose Keypoints"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        # Run pose inference on the frame
        results = model(frame, verbose=False)

        # Each result corresponds to this frame (list of len 1 typically)
        for result in results:
            if result.keypoints is None:
                continue
            # keypoints.xy shape -> (num_persons, num_keypoints, 2)
            keypoints_xy = result.keypoints.xy.cpu().numpy()
            # keypoints.conf shape -> (num_persons, num_keypoints)
            # print("keypoint:", keypoints_xy)
            keypoints_conf = None
            if hasattr(result.keypoints, "conf") and result.keypoints.conf is not None:
                keypoints_conf = result.keypoints.conf.cpu().numpy()
            draw_selected_keypoints(frame, keypoints_xy, keypoints_conf)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to quit
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
