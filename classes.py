from subprocess import run

def shell_run(input: str):
    return run(input, shell=True, capture_output=True, text=True)

class NoDevicesError(Exception):
    def __init__(self, message):
        super().__init__(message)

    def __str__(self):
        return self.message 

class ADBManager():
    def __init__(self):
        devices = shell_run("adb devices")
        if len(devices.stdout.strip().splitlines()) <= 1:
            raise NoDevicesError("No andriod emulators detected.")

    def _jump(self):
        shell_run("adb shell input swipe 540 1200 540 600 200")

    def _duck(self):
        shell_run("adb shell input swipe 540 600 540 1200 200")

    def _left(self):
        shell_run("adb shell input swipe 800 960 200 960 200")

    def _right(self):
        shell_run("adb shell input swipe 200 960 800 960 200")

