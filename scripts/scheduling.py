#!/usr/bin/env python3

import pathing

class Job(object):
    def __init__(self, pickup_pos, dropoff_pos):
        self.pickup_pos = pickup_pos
        self.dropoff_pos = dropoff_pos

        # The cached cost is the most recent cost calculation for a given robot
        # position. Reset to "None" when it may be inaccurate.
        self.cached_pos = None
        self.cached_cost = None

    def get_cost(self, maze, current_pos):
        if self.cached_pos == current_pos:
            return self.cached_cost
        else:
            self.cached_pos = current_pos
            path_a, path_b = self.execute_pathing(maze, current_pos)

            """
            Subtract two from the length of the path to account for the starting
            position in each path.
            """
            self.cached_cost = len(path_a) + len(path_b) - 2
            return self.cached_cost

    def execute_pathing(self, maze, current_pos):
        current_to_pickup = pathing.find_path_a_star(maze, current_pos, self.pickup_pos)
        pickup_to_dropoff = pathing.find_path_a_star(maze, self.pickup_pos, self.dropoff_pos)
        return current_to_pickup, pickup_to_dropoff

class Scheduling(object):
    def __init__(self, maze):
        self.jobs = []
        self.maze = maze

    def sort(self, current_pos):
        self.jobs.sort(key = lambda x: x.get_cost(self.maze, current_pos), reverse = False)

    def add_job(self, job: Job):
        self.jobs.append(job)

    def pop_job(self):
        if len(self.jobs) == 0:
            return None
        
        temp_job = self.jobs[0]
        del self.jobs[0]
        return temp_job

    def __str__(self):
        tmp_str = ""
        for i, j in enumerate(self.jobs):
            tmp_str += "Job #" + str(i)
            tmp_str += ", Pickup Pos: " + str(j.pickup_pos)
            tmp_str += ", Dropoff Pos: " + str(j.dropoff_pos)
            tmp_str += ", Cost: " + str(j.cached_cost)  + "\n"
        return tmp_str
        
    
if __name__=="__main__":
    maze1 = [[0, 0, 0], [1, 1, 0], [1, 1, 0]]

    sch = Scheduling(maze1)
    sch.add_job(Job((0, 2), (2, 2)))
    sch.add_job(Job((2, 2), (0, 0)))
    sch.add_job(Job((0, 0), (2, 2)))

    # cache paths + cost
    for j in sch.jobs:
        j.get_cost(sch.maze, (0, 0))

    # The costs, unsorted
    print(sch)

    # The costs, sorted at a current_pos of (0, 0). Because the paths are
    # cached, recalculation isn't needed.
    sch.sort((0, 0))
    print(sch)

    # The costs change after sorting at a specific current_pos
    sch.sort((0, 2))
    print(sch)
    
