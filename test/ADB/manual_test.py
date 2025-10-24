#%%
from pos2key.adb import SubwaySurfer, Grid
from time import sleep

#           Left   Centre   Right
# Jump    (-1,  1) (0,  1) (1,  1)
# Neutral (-1,  0) (0,  0) (1,  0)
# Roll    (-1, -1) (0, -1) (1, -1)
#
#       X  Y
#     (-1, 1)

test = SubwaySurfer(Grid.CENTRE_NEUTRAL.value)

#%%
test.move_to(Grid.LEFT_ROLL.value)
test.move_to(Grid.CENTRE_JUMP.value)
test.move_to(Grid.LEFT_ROLL.value)
test.move_to(Grid.CENTRE_JUMP.value)
test.move_to(Grid.LEFT_ROLL.value)
test.move_to(Grid.CENTRE_JUMP.value)
test.move_to(Grid.LEFT_ROLL.value)
test.move_to(Grid.CENTRE_JUMP.value)
test.move_to(Grid.LEFT_ROLL.value)
test.move_to(Grid.CENTRE_JUMP.value)

# from subprocess import run, CompletedProcess

# def shell_run(input: str) -> CompletedProcess:
#     """
#     Wrapper for subprocess.run(), so that its arguements can be centrally controlled (and code looks cleaner cause DRY).
#     """
#     return run(input, shell=True, capture_output=True, text=True)

# test = shell_run(r"ADB\adb devices")
# print(test.stderr, test.stdout)
# %%
