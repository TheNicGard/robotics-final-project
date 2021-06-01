import heapq
import math
from pprint import pprint

ALGO_TYPE = "euclidean"

"""
A* (pronounced "a star") works by modifying Dijkstra's algorithm to find a likely candidate for the
shortest path from a start node to a destination node. The algorithm is non-optimal; that is, it uses
a heuristic to guess which path is the shortest without checking every possible path. The downside to 
this is that A* will not return the shortest path every time it is run. The upside to this is that 
A* runs incredibly efficiently, checking only a fraction of all possible paths. Additionally, if your
heuristic is good, A* will still return the shortest path most of the time (or at least a path that
is acceptably short).



A* works by keeping an "open" and "closed" list. The closed list is essentially all the nodes the
algorithm has checked so far. The open list is all of the nodes the algorithm will check in the future.
Note that this is NOT all of the nodes that the algorithm has yet to check - it is really only the
unchecked "child" nodes of previously checked nodes. Nodes that are not direct neighbors to a checked
node during iteration i will not be in either list for iteration i. 



During each iteration, the algorithm uses an f-value to choose the next likely candidate. f is derived
from adding the cost to reach that node so far (g) and the estimate cost (h) of the remaining path derived
from the heuristic function. So f = g + h. The algorithm select the node in the open list with the lowest
f value when selecting a new node. If this new node is already in the closed list, the algorithm will remove 
this node from the open list and immediately select a different node. The algorithm uses the concept of 
"parenthood" to keep track of a path; when a node N is selected, the previous node M along that path is set 
as that node's parent. If at some later point, a new node O is discovered with a path to N that is better 
than the path through M, then N's parent is reset to O. 



We use a min-heap as our open list so that the root node is always guaranteed to be the lowest cost node in
the open list. This is better than a sorted list because A* tends to push 2-3 times more nodes into the 
open list than it pops, as each node will have between 1 and 4 neighbor nodes. Since it is best to minimize 
the common case, we use a heap which has O(log(n)) push and O(log(n)) pop. This is opposed to a sorted list
which has O(n) push and O(1) pop. 



The algorithm stops when the destination node is chosen as the node with the lowest f value, which will happen
as soon as it is bordering any node in the closed list. The path from the starting node to the destination node
is then derived by back-tracing the sequence of nodes via their parent field. 



Note again that this is not the provably shortest path between the two nodes. This algorithm doesn't fare 
particularly well when the shortest paths end up having a high up-front cost, i.e. it is greedy. But given
that our neighborhood in Gazebo is essentially grid-mappable (all the road sequences are the same length and
only stretch in NSEW directions), this should not be a big problem. Thus A* should have high accuracy for our
robot.



We use Euclidean distance as our heuristic function: dist = |x1 - x2| + |y1 - y2|. This is because we never want
the heuristic to over-estimate, but under-estimation is OK. Over-estimation would cause us to skip nodes that we
would want to consider, while under-estimation just slightly increases the nodes that you consider. Since the robot
can only move NSEW, a Euclidean distance is guaranteed to return a heuristic guess that is less than or equal to
the distance of the true shortest path. We originally tried Manhattan distance, but for some reason this took 
orders of magnitude longer. We are not sure why.
"""


class AStarNode(object):
    """ Node for A* algorithm """
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.f = 0
        self.g = 0
        self.h = 0

    # Overload equality function for easy checking
    def __eq__(self, comparator):
        return self.position == comparator.position

    # Define overloaded functions so we can use Node in heap
    def __gr__(self, comparator):
        return self.f > comparator.f 

    def __lt__(self, comparator):
        return self.f < comparator.f

    # Give unambiguous string reference
    def __repr__(self):
        return f"[{self.position}] - f({self.f}) = g({self.g}) + h({self.h})"


def heuristic_distance(x_node, y_node):
    """ Calculate Euclidean distance for two nodes """
    if ALGO_TYPE == "euclidean":
        return math.sqrt(((x_node.position[0] - y_node.position[0]) ** 2) + ((x_node.position[1] - y_node.position[1]) ** 2))
    elif ALGO_TYPE == "manhattan":
        return abs(x_node.position[0] - y_node.position[0]) + abs(x_node.position[1] - y_node.position[1])

def find_path_a_star(map, start_pos, dest_pos):
    """ Use A* algorithm to find a shortest path between start_pos and dest_pos """
    # Initialize nodes
    start_node = AStarNode(position=start_pos)
    dest_node = AStarNode(position=dest_pos)

    # Initialize open and closed lists
    open_list = []
    closed_list = []
    heapq.heapify(open_list)
    heapq.heappush(open_list, start_node)

    # Define neighbor nodes
    neighbor_positions = ((0, -1), (0, 1), (-1, 0), (1, 0),)

    # Iterate until destination 
    num_iterations = 0

    current_node = start_node
    while current_node != dest_node:
        # If open_list is empty, then there is no possible path
        if not open_list:
            print("ERROR: find_path_a_star: no possible path from", current_node.position, "to", dest_node.position)
            return None

        # Select current node for currenet iteration
        current_node = heapq.heappop(open_list)
        num_iterations += 1

        debug = False
        trace = False
        if debug:
            print("[" + str(num_iterations) + "]", "Using", str(current_node), ". Searching for", str(dest_node))

        # Add current node to closed list
        closed_list.append(current_node)

        # Select neighbor nodes
        neighbors = []
        for pos in neighbor_positions:
            neighbor_pos = (current_node.position[0] + pos[0], current_node.position[1] + pos[1])

            # Skip if position is off the map
            if (neighbor_pos[0] < 0 or neighbor_pos[1] < 0):
                if debug and trace: print("removing bc negative", neighbor_pos)
                continue
            if (neighbor_pos[0] >= len(map) or neighbor_pos[1] >= len(map[0])):
                if debug and trace: print("removing bc large", neighbor_pos, [len(map), len(map[0])])
                continue

            # Or if position is an obstacle
            if (map[neighbor_pos[0]][neighbor_pos[1]] != 0):
                if debug and trace: print("removing bc obstacle", neighbor_pos)
                continue

            # Add neighbor node and set parent to current node
            neighbor = AStarNode(parent=current_node, position=neighbor_pos)
            neighbors.append(neighbor)

        # Add neighbors to open list
        for neighbor in neighbors:
            # Make sure neighbor has not already been checked. We can do this bc of overloaded == statement :D
            if neighbor in closed_list:
                if debug and trace: print("skipping bc already in closed list", str(neighbor))
                continue

            # Initialize neighbor; since all paths are same length we just assign their weight a value of 1
            neighbor.h = heuristic_distance(neighbor, dest_node)
            neighbor.g = current_node.g + 1
            neighbor.f = neighbor.g + neighbor.h

            # Make sure neighbor hasn't already been selected for a better g value
            duplicates = [node for node in open_list if node == neighbor and node.g < neighbor.g]
            if len(duplicates) > 0:
                if debug and trace: print("skippping bc better candidate has already been found", str(neighbor))
                continue

            # "Oh won't you be my neighbor?!"
            heapq.heappush(open_list, neighbor)

        if debug:
            pprint(open_list)
            print("")

    print("Found path after", num_iterations, "iterations.")

    # Now we must backtrace the path
    best_path = []
    back_node = current_node
    while back_node:
        best_path.insert(0, back_node.position) # prepend
        back_node = back_node.parent

    print("Path is", len(best_path) - 1, "edges long.")
    print("Heuristic (" + ALGO_TYPE + ") would predict path of", heuristic_distance(start_node, dest_node), "edges long")
    return best_path

def example(print_maze = True):

    maze1 = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,] * 2,
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,] * 2,
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,] * 2,
            [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,] * 2,
            [0,0,0,1,1,0,0,1,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,] * 2,
            [0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,] * 2,
            [0,0,0,1,0,1,1,1,1,0,1,1,0,0,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,] * 2,
            [0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,1,1,0,1,0,0,0,0,0,0,1,1,1,0,] * 2,
            [0,0,0,1,0,1,1,0,1,1,0,1,1,1,0,0,0,0,0,1,0,0,1,1,1,1,1,0,0,0,] * 2,
            [0,0,0,1,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,0,1,0,1,1,] * 2,
            [0,0,0,1,0,1,0,1,1,0,1,1,1,1,0,0,1,1,1,1,1,1,1,0,1,0,1,0,0,0,] * 2,
            [0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,1,0,] * 2,
            [0,0,0,1,0,1,1,1,1,0,1,0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,1,0,0,0,] * 2,
            [0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,1,] * 2,
            [0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,] * 2,
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,] * 2,]

    maze2 = [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,1],
        [0,1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0],
        [0,0,1,0,1,0,1,0,1,0,1,1,0,0,0,0,0,1,0,0,0,1,1,0,1,1,1,1,1,1,1,0],
        [0,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,0,1,0,1,1,1,1,1,1],
        [0,1,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,0,1,0,1,0,0,0,0,0,0,0],
        [0,1,0,1,1,1,1,1,0,0,0,0,1,0,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,0],
        [0,1,0,1,0,0,0,0,0,1,1,0,0,0,1,0,0,0,1,1,1,0,1,0,1,0,0,0,0,1,0,0],
        [0,1,0,1,0,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,0,0,0,0,0,1,1,0,1,1,0],
        [0,1,0,1,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,0,1,0,0],
        [0,1,0,1,0,0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,1,1,1,0,1,1,0,1,0,1,1,0],
        [0,1,0,1,0,1,1,1,0,1,0,1,1,0,1,1,0,0,0,0,0,1,1,0,1,1,0,1,0,1,0,0],
        [0,1,0,1,0,0,0,1,0,0,0,1,1,0,1,1,0,1,1,1,0,0,0,0,1,1,0,1,0,1,1,0],
        [0,1,0,1,0,1,1,1,0,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,0,0],
        [0,1,0,1,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,1,1],
        [0,1,0,0,0,0,0,0,0,1,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1],
        [0,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,1,1,0,0,0,0,0,0,1,0,1,0,1,0,1],
        [0,1,0,1,1,1,1,1,1,1,1,0,1,0,1,0,1,1,1,1,1,1,1,1,0,1,0,1,0,0,0,0],
        [0,1,0,1,0,1,0,0,1,0,0,0,1,0,1,0,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1],
        [0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0,0,0,0,0,0,0,1,1,0,1,0,1,0,1,0,1],
        [0,1,0,1,0,1,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1,0,1,1,0,1,0,1,0,1,1,1],
        [0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ]

    maze3 = [
        [0,0,0,0,0,0,0,0,0],
        [0,1,0,1,0,1,1,1,0],
        [0,0,0,0,0,0,0,0,0],
        [0,1,0,1,0,1,1,1,0],
        [0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,0]
    ]

    maze4 = [
        [0,0,0,0,0,0,0,0,0],
        [0,1,0,1,0,1,0,1,0],
        [0,0,0,0,0,0,0,0,0],
        [0,1,0,1,0,1,0,1,0],
        [0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1],
        [0,0,0,0,0,0,0,0,0]
    ]

    maze = maze2
    
    start = (0, 0)
    end = (len(maze)-1, len(maze[0])-1)

    path = find_path_a_star(maze, start, end)

    if print_maze:
        if path:
            for step in path:
                maze[step[0]][step[1]] = 2
      
        border = "\u2588" * (len(maze[0]) + 2) * 2
        print(border)
        skip = False
        row_num = 0
        for row in maze:
            line = ["\u2588\u2588"]
            for col in row:
                # if (row_num, col) == start: line = ["  "]
                if col == 1:
                    line.append("\u2588\u2588")
                elif col == 0:
                    line.append("  ")
                elif col == 2:
                    line.append(". ")
                # if (row_num, col) == end: 
                #     line.append(" \u2588")
                #     skip = True
            if not skip:
                line.append("\u2588\u2588")
            print("".join(line))
            row_num += 1
        print(border)

    # print(path)
    print("")

if __name__ == "__main__":
    example()
    ALGO_TYPE = "manhattan"
    example()