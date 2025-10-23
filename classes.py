from subprocess import run
from enum import Enum

def shell_run(input: str):
    return run(input, shell=True, capture_output=True, text=True)

class ADBError(Exception):
    __module__ = "__main__"

    def __init__(self, message):
        super().__init__(message)

class ADBManager():
    def __init__(self):
        connection = shell_run("adb connect 127.0.0.1:5555")
        if "No connection could be made" in connection.stdout.strip():
            raise ADBError(connection.stdout)

        devices = shell_run("adb devices")
        if len(devices.stdout.strip().splitlines()) <= 1:
            raise ADBError("No android emulators detected.")

    def _jump(self):
        return shell_run("adb shell input swipe 540 1200 540 600 200")

    def _roll(self):
        return shell_run("adb shell input swipe 540 600 540 1200 200")

    def _left(self):
        return shell_run("adb shell input swipe 800 960 200 960 200")

    def _right(self):
        return shell_run("adb shell input swipe 200 960 800 960 200")


class Grid(Enum):
    LEFT_JUMP = {"x": -1, "y": 1}
    CENTRE_JUMP = {"x": 0, "y": 1}
    RIGHT_JUMP = {"x": 1, "y": 1}
    LEFT_NEUTRAL = {"x": -1, "y": 0}
    CENTRE_NEUTRAL = {"x": 0, "y": 0}
    RIGHT_NEUTRAL = {"x": 1, "y": 0}
    LEFT_ROLL = {"x": -1, "y": -1}
    CENTRE_ROLL = {"x": 0, "y": -1}
    RIGHT_ROLL = {"x": 1, "y": -1}

class SubwaySurfer(ADBManager):
    def __init__(self, position=Grid.CENTRE_NEUTRAL.value):
        super().__init__()
        self.x = position["x"]
        self.y = position["y"]

    def _moveX(self, x_distance):
        return self._right() if x_distance >= 0 else self._left()
    
    def _moveY(self, x_distance):
        return self._jump() if x_distance >= 0 else self._roll()

    def move_to(self, desired_position):
        """
                  Left   Centre   Right
        Jump    (-1,  1) (0,  1) (1,  1)
        Neutral (-1,  0) (0,  0) (1,  0)
        Roll    (-1, -1) (0, -1) (1, -1)

              X  Y
            (-1, 1)
        """
    
        x_distance_to_travel = desired_position["x"] - self.x
        y_distance_to_travel = desired_position["y"] - self.y

        print(f"X Offset: {x_distance_to_travel}, Y Offset: {y_distance_to_travel}")

        for _ in range(x_distance_to_travel):
            self._moveX(x_distance_to_travel)

        for _ in range(y_distance_to_travel):
            self._moveY(y_distance_to_travel)
