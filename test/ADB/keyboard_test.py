# Chatgpt created this btw
# pip install keyboard

import keyboard
import time
from pos2key.adb import SubwaySurfer

# debounce interval in seconds (avoid firing repeatedly while holding)
DEBOUNCE = 0.15
_last_called = {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0}
jake = SubwaySurfer()

def _handle_event(ev: keyboard.KeyboardEvent):
    # We only want key-down events (not key-up)
    if ev.event_type != "down":
        return

    name = ev.name  # 'up', 'down', 'left', 'right', etc.
    if name not in _last_called:
        return

    now = time.time()
    if now - _last_called[name] < DEBOUNCE:
        return  # too soon, ignore (debounce)
    _last_called[name] = now

    if name == "up":
        print(jake._moveY(1)) # Equivalent to _jump()
    elif name == "down":
        print(jake._moveY(-1)) # Equivalent to _roll()
    elif name == "left":
        print(jake._moveX(-1)) # Equivalent to _left()
    elif name == "right":
        print(jake._moveX(1))# Equivalent to _right()
    print(f"{name} pressed")

def main():
    print("Arrow listener running. Press arrow keys (↑ ↓ ← →). Press ESC to quit.")
    # Register the global handler (non-blocking)
    keyboard.hook(_handle_event)

    # Wait until user presses ESC (this is a blocking wait but does not prevent handlers from running)
    keyboard.wait("esc")
    print("Exiting...")

if __name__ == "__main__":
    main()
