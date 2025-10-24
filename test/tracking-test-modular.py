"""
Input 1 to start live tracking after all models are loaded to start the tracking
Tracking auto reselects using the depth scan when the target is lost.
"""
from pos2key.tracking import Tracker

tracker = Tracker()

while True:
    choice = input(">>>")
    if choice == "1":
        tracker.begin_tracking(show_other_dets=True)
    else:
        exit()