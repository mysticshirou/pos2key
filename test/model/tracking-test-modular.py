"""
Input 1 to start live tracking after all models are loaded to start the tracking
Tracking auto reselects using the depth scan when the target is lost.
"""
from pos2key.tracking import Tracker

tracker = Tracker()
tracker.set_grid_offset((0.5, 0.5), (0.5, 0.5))
tracker.set_model_path("./models/yolo11n.pt")

while True:
    choice = input(">>>")
    if choice == "1":
        tracker.begin_tracking(broadcast_fn=print, show_other_dets=True, use_wayland_viewer=True)
    else:
        exit()