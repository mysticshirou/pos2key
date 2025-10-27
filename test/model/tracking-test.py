"""
Input 1 to start live tracking after all models are loaded to start the tracking
Tracking auto reselects using the depth scan when the target is lost.
"""

import cv2
import time
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline
import os
import shlex
import subprocess

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
model = YOLO("yolo11n.pt")

def begin_videocapture():
    cap = cv2.VideoCapture(0)

    GRID_OFFSETX = 100
    GRID_OFFSETY = 50
    GRID_HORIZONTAL = ()  # Horizontal threshold
    GRID_VERTICAL = ()    # Vertial threshold
    TRACKING_ID = None      # Tracking ID of object

    BBOX_COLOUR = (0, 255, 0)
    GRID_COLOUR = (0, 0, 255)

    def draw_gridlines(image, h_thresh: tuple[int], v_thresh: tuple[int]):
        image = cv2.line(image, (h_thresh[0], 0), (h_thresh[0], image.shape[0]), GRID_COLOUR, 2)
        image = cv2.line(image, (h_thresh[1], 0), (h_thresh[1], image.shape[0]), GRID_COLOUR, 2)
        image = cv2.line(image, (0, v_thresh[0]), (image.shape[1], v_thresh[0]), GRID_COLOUR, 2)
        image = cv2.line(image, (0, v_thresh[1]), (image.shape[1], v_thresh[1]), GRID_COLOUR, 2)
        return image

    class FrameViewer:
        def __init__(self):
            self.backend = "cv2"
            if os.environ.get("WAYLAND_DISPLAY") or os.environ.get("USE_WAYLAND_VIEWER"):
                self.backend = "ffplay"
            self.process = None
            self.width = None
            self.height = None
            self.fps = 30
            self.window_name = "Camera"

        def open(self, width: int, height: int, fps: int = 30, window_name: str = "Camera"):
            self.width = width
            self.height = height
            self.fps = fps
            self.window_name = window_name
            if self.backend != "ffplay":
                return

            cmd = (
                f"ffplay -f rawvideo -pixel_format bgr24 -video_size {width}x{height}"
                f" -framerate {fps} -window_title {shlex.quote(window_name)} -i - -hide_banner -loglevel error"
            )
            try:
                # Start ffplay
                self.process = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                print("ffplay not found; falling back to cv2.imshow")
                self.backend = "cv2"

        def show(self, frame, window_name: str = "Camera"):
            if self.backend == "cv2":
                cv2.imshow(window_name, frame)
                return

            # backend is ffplay
            if self.process is None or self.process.poll() is not None:
                # (re)open
                self.open(frame.shape[1], frame.shape[0], self.fps, window_name)
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                try:
                    if self.process:
                        self.process.kill()
                except Exception:
                    pass
                self.open(frame.shape[1], frame.shape[0], self.fps, window_name)

            try:
                if self.process and self.process.stdin:
                    self.process.stdin.write(frame.tobytes())
                    self.process.stdin.flush()
                else:
                    # fallback
                    cv2.imshow(window_name, frame)
            except Exception:
                cv2.imshow(window_name, frame)

        def close(self):
            if self.backend == "cv2":
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
                return

            if self.process:
                try:
                    if self.process.stdin:
                        self.process.stdin.close()
                except Exception:
                    pass
                try:
                    self.process.terminate()
                    self.process.wait(timeout=1)
                except Exception:
                    try:
                        self.process.kill()
                    except Exception:
                        pass

    start = True
    viewer = None
    while cap.isOpened():
        ret, frame = cap.read()
        if start: ret, frame = cap.read()
        if not ret:
            break

        if start:
            # Initialise video_writer
            video_writer = cv2.VideoWriter('tracking_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame.shape[1], frame.shape[0]))

            # Initialise viewer
            try:
                viewer = FrameViewer()
                viewer.open(frame.shape[1], frame.shape[0], fps=30, window_name='Camera')
            except Exception as e:
                print(f"Viewer init failed: {e}")

            start_time = time.perf_counter()
            # Depth scan initial frame for gridlines
            # Convert the cv2 image into PIL for depth scan
            scan_frame = frame.copy()
            scan_frame_pil = Image.fromarray(cv2.cvtColor(scan_frame, cv2.COLOR_BGR2RGB))

            depth = pipe(scan_frame_pil)["depth"]
            cvdepth = cv2.cvtColor(np.array(depth), cv2.COLOR_RGB2BGR)

            # K means clustering to create segmentation mask
            pixel_values = cvdepth.reshape((-1, 3))
            pixel_values = np.float32(pixel_values)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            compactness, labels, (centers) = cv2.kmeans(pixel_values, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            centers = np.uint8(centers)
            clustered_frame = centers[labels.flatten()]
            clustered_frame = clustered_frame.reshape(cvdepth.shape)
            _, binary_image = cv2.threshold(cv2.cvtColor(clustered_frame, cv2.COLOR_BGR2GRAY), np.max(clustered_frame)-1, 255, cv2.THRESH_BINARY)

            masked_frame = cv2.bitwise_and(scan_frame, scan_frame, mask=binary_image)
            masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)

            # Find and track main target
            results = model.track(masked_frame, persist=True, conf=0.1)
            for det in results[0].boxes:
                if int(det.cls) == 0:    # If person class (0 in this case)
                    x1, y1, x2, y2 = map(int, det.xyxy[0])
                    TRACKING_ID = int(det.id) if det.id is not None else -1  # Track ID
                    conf = det.conf.item()      # Confidence score
                    center = (int((x2-x1) / 2 + x1), int((y2-y1) / 2 + y1))

                    # Bbox
                    alpha = 0.4
                    scan_frame = cv2.rectangle(scan_frame, (x1, y1), (x2, y2), BBOX_COLOUR, 2)
                    overlay = frame.copy()
                    overlay = cv2.rectangle(overlay, (x1, y1), (x2, y2), BBOX_COLOUR, -1)
                    scan_frame = cv2.addWeighted(overlay, alpha, scan_frame, 1 - alpha, 0.0)
                    # Bbox Center
                    scan_frame = cv2.circle(scan_frame, center, 5, BBOX_COLOUR, 2)
                    # Label
                    label = f'ID: {TRACKING_ID} | Conf: {conf:.2f}'
                    cv2.putText(scan_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GRID_COLOUR, 2)
                    # Grid lines
                    GRID_HORIZONTAL = (center[0] + GRID_OFFSETX, center[0] - GRID_OFFSETX)
                    GRID_VERTICAL = (center[1] + GRID_OFFSETY, center[1] - GRID_OFFSETY)

                    scan_frame = draw_gridlines(scan_frame, GRID_HORIZONTAL, GRID_VERTICAL)
                    break

            cv2.imwrite("initial_scan.png", scan_frame)

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"Depths scan runtime: {elapsed_time:.6f} seconds")
            start = False
            continue

        # Run YOLO tracking on the frame
        results = model.track(frame, persist=True, conf=0.3, iou=0.5)
        annotated_frame = frame.copy()
        annotated_frame = draw_gridlines(annotated_frame, GRID_HORIZONTAL, GRID_VERTICAL)

        id_found = False
        for det in results[0].boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            if det.id == TRACKING_ID:
                id_found = True
                center = (int((x2-x1) / 2 + x1), int((y2-y1) / 2 + y1))
                conf = det.conf.item()

                # Bbox
                alpha = 0.4
                annotated_frame = cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), BBOX_COLOUR, 2)
                overlay = frame.copy()
                overlay = cv2.rectangle(overlay, (x1, y1), (x2, y2), BBOX_COLOUR, -1)
                annotated_frame = cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0.0)
                # Bbox Center
                annotated_frame = cv2.circle(annotated_frame, center, 5, BBOX_COLOUR, 2)
                # Label
                label = f'ID: {TRACKING_ID} | Conf: {conf:.2f}'
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GRID_COLOUR, 2)
                # Grid lines
            else:
                conf = det.conf.item()
                # Bbox
                annotated_frame = cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (100, 0, 0), 1)
                # Label
                label = f'ID: {det.id} | Conf: {conf:.2f}'
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GRID_COLOUR, 1)

        # Save the frame
        if not id_found:
            print("Lost track of person, rescanning...")
            start = True

        video_writer.write(annotated_frame)
        print("Written to video writer")

        # Show with viewer (falls back to cv2.imshow when appropriate)
        if viewer is not None:
            viewer.show(annotated_frame, window_name='Camera')
        else:
            cv2.imshow('Camera', annotated_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    video_writer.release()
    # Close viewer if used
    try:
        if viewer is not None:
            viewer.close()
    except Exception:
        pass

while True:
    choice = int(input(">>>"))
    if choice == 1:
        begin_videocapture()
    else:
        exit()