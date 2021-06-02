import numpy as np
import os

PATH_PREFIX = os.path.dirname(__file__) + "/robot_maps/"


if __name__ == "__main__":
    neighborhood_simple = [
        [0,0,0,0,0,0,0,0,0],
        [0,1,0,1,0,1,0,1,0],
        [0,1,0,1,0,0,0,0,0],
        [0,1,0,1,0,1,0,1,0],
        [0,1,0,1,0,0,0,0,0],
        [0,1,0,1,0,1,0,1,0],
        [0,1,0,1,0,1,0,1,0],
        [0,1,0,1,0,1,0,1,0],
        [0,0,0,0,0,0,0,0,0],
    ]

    np.savetxt(PATH_PREFIX + "neighborhood_simple.txt", neighborhood_simple)