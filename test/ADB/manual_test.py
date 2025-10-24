#%%
from pos2key.classes import SubwaySurfer, Grid

#           Left   Centre   Right
# Jump    (-1,  1) (0,  1) (1,  1)
# Neutral (-1,  0) (0,  0) (1,  0)
# Roll    (-1, -1) (0, -1) (1, -1)
#
#       X  Y
#     (-1, 1)

test = SubwaySurfer(Grid.CENTRE_NEUTRAL.value)

#%%

#%%
while True:
    test.moveX(1)
#     test._jump()
#     test._roll()
#     test._left()
#     test._right()