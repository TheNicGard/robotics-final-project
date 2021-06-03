import numpy as np
import json
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

    neighborhood_simple_colors = {
        "pickup": {
            "magenta": [1, 0],
            "blue": [1, 2],
            "red": [1, 6],
            "green": [3, 6],
            "orange": [5, 0],
            "cyan": [7, 6]
        },
        "dropoff": {
            "magenta": [1, 2],
            "blue": [1, 4],
            "red": [1, 8],
            "green": [3, 8],
            "orange": [5, 2],
            "cyan": [7, 8]
        }
    }

    np.savetxt(PATH_PREFIX + "neighborhood_simple.txt", neighborhood_simple)
    with open(PATH_PREFIX + "neighborhood_simple.json", "w") as outfile:
        json.dump(neighborhood_simple_colors, outfile)

