import os
import cv2
import time
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from transformers import pipeline

from pos2key.config import Config
cfg = Config()

def draw_gridlines(image, h_thresh: list[int]|tuple[int], v_thresh: list[int]|tuple[int], colour=(0,0,255)):
    """Draws gridlines on image"""
    image = cv2.line(image, (h_thresh[0], 0), (h_thresh[0], image.shape[0]), colour, 2)
    image = cv2.line(image, (h_thresh[1], 0), (h_thresh[1], image.shape[0]), colour, 2)
    image = cv2.line(image, (0, v_thresh[0]), (image.shape[1], v_thresh[0]), colour, 2)
    image = cv2.line(image, (0, v_thresh[1]), (image.shape[1], v_thresh[1]), colour, 2)
    return image

def on_event(event):
    # Does whatever 
    print(event)

class Tracker:
    def __init__(self, bbox_colour = (0,255,0), grid_colour=(0,0,255), camera_id=0, cls=0):
        self.tracking_model = YOLO(cfg.get("yolo_model_path", default="models/yolo11n.pt"))
        self.depth_model = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

        self.GRID_OFFSETX = (100, -100)
        self.GRID_OFFSETY = (50, -50)

        self.BBOX_COLOUR = bbox_colour
        self.GRID_COLOUR = grid_colour
        self.CAMERA = camera_id         # Camera index to use, default is 0
        self.PERSON = cls               # Class # for person class, default is 0

        self.output_dir = os.path.join(os.getcwd(), "outputs")

    def set_model_path(self, model_path: Path):
        """
        Changes yolo_model_path in config and reloads the tracking model used

        model_path: path to YOLO model
        """
        cfg.set("yolo_model_path", model_path)
        self.tracking_model = YOLO(cfg.get("yolo_model_path", default="models/yolo11n.pt"))
        return 1
    
    def set_grid_offset(self, offsetx: tuple[int|float] = (100, 100), offsety: tuple[int|float] = (50, 50)):
        """
        Set the grid offsets by any numerical value
        """
        self.GRID_OFFSETX = (offsetx[0], offsetx[1] if offsetx[1] <= 0 else -offsetx[1])
        self.GRID_OFFSETY = (offsety[0], offsety[1] if offsety[1] <= 0 else -offsety[1])
        return 1
    
    def depth_scan(self, frame: np.array):
        """
        Uses the Depth-Anything-V2-Small model to determine closest objects to camera

        Inputs:
            frame: Current frame of the cv2.VideoCapture

        Outputs:
            masked_frame: Final segmented image of closest objects to the camera, created using binary_image mask & initial frame
            clustered_frame: Depth scan image segmented using k-means clustering, for debugging
            binary_image: Thresholded clustered_frame to create binary image mask, for debugging
        """
        # Convert the cv2 image into PIL for depth scan
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        depth = self.depth_model(frame_pil)["depth"]
        cvdepth = cv2.cvtColor(np.array(depth), cv2.COLOR_RGB2BGR)

        # K means clustering to create segmentation mask
        pixel_values = cvdepth.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, (centers) = cv2.kmeans(pixel_values, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        clustered_frame = centers[labels.flatten()]
        clustered_frame = clustered_frame.reshape(cvdepth.shape)
        _, binary_image = cv2.threshold(cv2.cvtColor(clustered_frame, cv2.COLOR_BGR2GRAY), np.max(clustered_frame)-1, 255, cv2.THRESH_BINARY)

        masked_frame = cv2.bitwise_and(frame, frame, mask=binary_image)
        masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
        return masked_frame, clustered_frame, binary_image

    def _check_position(self, position: tuple[int], h_thresh: list[int]|tuple[int], v_thresh: list[int]|tuple[int]):
        if position[0] > h_thresh[0]: on_event("left")
        elif position[0] < h_thresh[1]: on_event("right")
        else: on_event("middle")

        if position[1] > v_thresh[0]: on_event("crouch")
        elif position[1] < v_thresh[1]: on_event("jump")
        else: on_event("neutral")

        print("")

    def begin_tracking(self, save=True, show_other_dets=False, fps=30, verbose=False):
        """Starts real time tracking"""
        cap = cv2.VideoCapture(self.CAMERA)
        ret, frame = cap.read()
        assert ret
        video_writer = cv2.VideoWriter(os.path.join(self.output_dir, "tracking_output.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))
        do_depth_scan = True
        
        while cap.isOpened():
            _, frame = cap.read()
            annotated_frame = frame.copy()

            if do_depth_scan:
                s = time.perf_counter()
                segmented, _, _ = self.depth_scan(annotated_frame)
                results = self.tracking_model.track(segmented, persist=True, conf=0.1, verbose=verbose)

                for det in results[0].boxes:
                    if int(det.cls) == self.PERSON:    # If person class (0 in this case)
                        x1, y1, x2, y2 = map(int, det.xyxy[0])
                        TRACKING_ID = int(det.id.item()) if det.id is not None else -1  # Track ID
                        conf = det.conf.item()      # Confidence score
                        center = (int((x2-x1) / 2 + x1), int((y2-y1) / 2 + y1))

                        annotated_frame = cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), self.BBOX_COLOUR, 2)     # Bounding box drawing
                        overlay = frame.copy()                                                                        # 
                        overlay = cv2.rectangle(overlay, (x1, y1), (x2, y2), self.BBOX_COLOUR, -1)                    # 
                        annotated_frame = cv2.addWeighted(overlay, 0.4, annotated_frame, 0.6, 0.0)                    #

                        annotated_frame = cv2.circle(annotated_frame, center, 5, self.BBOX_COLOUR, 2)                 # Player position

                        label = f'ID: {TRACKING_ID} | Conf: {conf:.2f}'                                               # Label
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.GRID_COLOUR, 2)

                        # Grid lines horizontal
                        GRID_HORIZONTAL, GRID_VERTICAL = [], []
                        for value in self.GRID_OFFSETX:
                            if isinstance(value, int):
                                GRID_HORIZONTAL.append(center[0] + value)
                            elif isinstance(value, float):
                                GRID_HORIZONTAL.append(center[0] + round(value * (x2-x1)))

                        # Grid lines vertical
                        for value in self.GRID_OFFSETY:
                            if isinstance(value, int):
                                GRID_VERTICAL.append(center[1] + value)
                            elif isinstance(value, float):
                                GRID_VERTICAL.append(center[1] + round(value * (y2-y1)))

                        annotated_frame = draw_gridlines(annotated_frame, GRID_HORIZONTAL, GRID_VERTICAL)
                        break
                
                if save: 
                    segmented = cv2.rectangle(segmented, (x1, y1), (x2, y2), self.BBOX_COLOUR, 2)
                    segmented = draw_gridlines(segmented, GRID_HORIZONTAL, GRID_VERTICAL)
                    cv2.imwrite(os.path.join(self.output_dir, "initial_scan.png"), annotated_frame)
                    cv2.imwrite(os.path.join(self.output_dir, "masked_scan.png"), segmented)
                    print(f"Saved to {os.path.join(self.output_dir, 'initial_scan.png')}")

                e = time.perf_counter()
                print(f"Depth scan runtime: {e-s:.6f} seconds")
                do_depth_scan = False
                continue

            # Run YOLO tracking on the frame
            results = self.tracking_model.track(frame, persist=True, conf=0.3, iou=0.5, verbose=verbose)
            annotated_frame = frame.copy()
            annotated_frame = draw_gridlines(annotated_frame, GRID_HORIZONTAL, GRID_VERTICAL)

            id_found = False
            for det in results[0].boxes:
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                if det.id == TRACKING_ID:
                    id_found = True
                    center = (int((x2-x1) / 2 + x1), int((y2-y1) / 2 + y1))
                    self._check_position(center, GRID_HORIZONTAL, GRID_VERTICAL)
                    conf = det.conf.item()

                    annotated_frame = cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), self.BBOX_COLOUR, 2)   # Bounding box drawing
                    overlay = frame.copy()                                                                      # 
                    overlay = cv2.rectangle(overlay, (x1, y1), (x2, y2), self.BBOX_COLOUR, -1)                  # 
                    annotated_frame = cv2.addWeighted(overlay, 0.4, annotated_frame, 0.6, 0.0)                  #

                    annotated_frame = cv2.circle(annotated_frame, center, 5, self.BBOX_COLOUR, 2)               # Player position

                    label = f'ID: {TRACKING_ID} | Conf: {conf:.2f}'                                             # Label
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.GRID_COLOUR, 2)

                elif show_other_dets:
                    conf = det.conf.item()
                    tracking_id = int(det.id.item()) if det.id is not None else -1
                    annotated_frame = cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (100, 0, 0), 1)        # Bounding box drawing

                    label = f'ID: {tracking_id} | Conf: {conf:.2f}'                                             # Label
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.GRID_COLOUR, 1)

            # Save the frame
            if not id_found:
                print("Lost track of person, rescanning...")
                do_depth_scan = True

            video_writer.write(annotated_frame)
            if verbose: print("Written to video writer")

            cv2.imshow('Camera', annotated_frame)

            if cv2.waitKey(1) == ord('q'):  # Press q to stop live tracking
                break

        cap.release()
        video_writer.release()


if __name__ == "__main__":
    tracker = Tracker()

    while True:
        choice = input(">>>")
        if choice == "1":
            tracker.begin_tracking(show_other_dets=True)
        else:
            exit()