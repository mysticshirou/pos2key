from subprocess import run, CompletedProcess
from enum import Enum
from time import sleep

class ADBError(Exception):
    # Makes error appear in main namespace (ADBError: <exception> vs classes.ADBError: <exception>)
    __module__ = "__main__"

    def __init__(self, error_message: str) -> None:
        # Prints error message received
        super().__init__(error_message)

class ADBManager():
    def __init__(self, adb_endpoint: str = "127.0.0.1:5555", adb_location: str = r"ADB\adb.exe") -> None:
        """
        Connects to the open ADB port with "adb connect <endpoint>".
        Checks if connection is successful with "adb devices".
        When both are successful, android emulator can be controlled.
        """
        self.adb_location = adb_location

        connection = self.shell_run(f"{self.adb_location} connect {adb_endpoint}")
        if "No connection could be made" in connection.stdout.strip():
            raise ADBError(connection.stdout)
        else: print(fr"ADB {connection.stdout}") # should see "ADB connected to 127.0.0.1:5555"

        devices = self.shell_run(f"{self.adb_location} devices")
        if len(devices.stdout.strip().splitlines()) <= 1:
            raise ADBError(f"No android emulators detected. \nstdout: {devices.stdout}")
        else: print("ADB ready")

    @staticmethod
    def shell_run(input: str) -> CompletedProcess:
        """
        Wrapper for subprocess.run(), so that its arguements can be centrally controlled (and code looks cleaner cause DRY).
        """
        return run(input, shell=True, capture_output=True, text=True)

    def _jump(self) -> CompletedProcess:
        return self.shell_run(f"{self.adb_location} shell input swipe 540 1200 540 600 200")

    def _roll(self) -> CompletedProcess:
        return self.shell_run(f"{self.adb_location} shell input swipe 540 600 540 1200 200")

    def _left(self) -> CompletedProcess:
        return self.shell_run(f"{self.adb_location} shell input swipe 800 960 200 960 200")

    def _right(self) -> CompletedProcess:
        return self.shell_run(f"{self.adb_location} shell input swipe 200 960 800 960 200")


class Grid(Enum):
    """
              Left   Centre   Right
    Jump    (-1,  1) (0,  1) (1,  1)
    Neutral (-1,  0) (0,  0) (1,  0)
    Roll    (-1, -1) (0, -1) (1, -1)

          X  Y
        (-1, 1)
    """

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
    def __init__(self, position: dict[str, int] = Grid.CENTRE_NEUTRAL.value) -> None:
        super().__init__()

        # Initialise default character position (In the centre column, neutral position = CENTRE_NEUTRAL)
        self.x = position["x"]
        self.y = position["y"]

    def _moveX(self, x_distance: int) -> CompletedProcess:
        """
        Determines the X direction by checking whether x_distance is positive or negative. 
        """
        print("right"  if x_distance >= 0 else "left")
        return self._right() if x_distance >= 0 else self._left()
    
    def _moveY(self, y_distance: int) -> CompletedProcess:
        """
        Determines the Y direction by checking whether x_distance is positive or negative. 
        """
        print("jump"  if y_distance >= 0 else "roll")
        return self._jump() if y_distance >= 0 else self._roll()

    def move_to(self, desired_position: Grid) -> None:
        """
                  Left   Centre   Right
        Jump    (-1,  1) (0,  1) (1,  1)
        Neutral (-1,  0) (0,  0) (1,  0)
        Roll    (-1, -1) (0, -1) (1, -1)

              X  Y
            (-1, 1)
        """
        print(f"Moving to {desired_position}")

        # Calculates X & Y distances to travel by finding the difference between the desired and current positions.
        x_distance_to_travel = desired_position["x"] - self.x
        y_distance_to_travel = desired_position["y"] - self.y
        print(f"Current Pos: {self.x, self.y}, Desired Pos: {desired_position}. X Offset: {x_distance_to_travel}, Y Offset: {y_distance_to_travel}")

        # Executes movements
        for _ in range(abs(x_distance_to_travel)):
            self._moveX(x_distance_to_travel)
            sleep(0.1)

        for _ in range(abs(y_distance_to_travel)):
            self._moveY(y_distance_to_travel)
            sleep(0.1)

        # Sets desired X position to be the new current positon.
        # Y positon does not need to be set as it auto resolves itself (Character will stop jumping / rolling automatically). 
        self.x = desired_position["x"]
