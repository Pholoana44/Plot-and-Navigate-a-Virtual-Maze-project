import numpy as np
import random
import sys
import time
import operator

class Robot(object):
    def __init__(self, maze_dim):
        #Direction dictionaries
        self.dir_sensors = {
            'u': ['l', 'u', 'r'],
            'r': ['u', 'r', 'd'],
            'd': ['r', 'd', 'l'],
            'l': ['d', 'l', 'u'],
            'up': ['l', 'u', 'r'],
            'right': ['u', 'r', 'd'],
            'down': ['r', 'd', 'l'],
            'left': ['d', 'l', 'u']
        }
        
        self.dir_move = {
            'u': [0, 1],
            'r': [1, 0],
            'd': [0, -1],
            'l': [-1, 0],
            'up': [0, 1],
            'right': [1, 0],
            'down': [0, -1],
            'left': [-1, 0]
        }
        
        self.dir_reverse = {
            'u': 'd',
            'r': 'l',
            'd': 'u',
            'l': 'r',
            'up': 'd',
            'right': 'l',
            'down': 'u',
            'left': 'r'
        }
        
        self.map_dic= {
            'up': 3,
            'right': 0,
            'down': 1,
            'left': 2,
            'u': 3,
            'r': 0,
            'd': 1,
            'l': 2
        }
        
        self.location = [0, 0]
        self.heading = 'up'
        self.maze_dim = maze_dim
        self.maze_area = maze_dim ** 2
        self.goal_bounds = [maze_dim / 2 - 1, maze_dim / 2]
        self.base_rotation = [-90, 0, 90]
        self.base_movement = [-3, -2, -1, 0, 1, 2, 3]

        # more information to learn
        self.run = 0
        self.move_time = 0
        self.goal_position = [0, 0]         # remember the goal_position
        self.dead_zone = False              # check if the robot in the dead_zone
        self.remember_goal = False          # check if the robot hit_goal
        self.last_move_backward = False     # check last step robot move_backward
        self.last_movement = 0
        self.path_length = 0

        self.map_location = [[' ' for row in range(maze_dim)] for col in range(maze_dim)]

        self.map_dead_zone = [[' ' for row in range(maze_dim)] for col in range(maze_dim)]

        self.map_count = [[0 for row in range(maze_dim)] for col in range(maze_dim)]
        self.map_count[0][0] = 1

        self.map_heuristic = [[min(abs(row-maze_dim/2+1), abs(row-maze_dim/2))+min(abs(col-maze_dim/2+1),
                            abs(col-maze_dim/2)) for row in range(maze_dim)] for col in range(maze_dim)]

        self.map_maze = [[[0,0,0,0] for row in range(maze_dim)] for col in range(maze_dim)]

        self.mapped_maze = [[15 for row in range(maze_dim)] for col in range(maze_dim)]

        self.value = [[99 for row in range(self.maze_dim)] for col in range(self.maze_dim)]

         # Set the exploration model
        self.exploration_model = 'heuristic'
        
        # Adjust flags based on the selected exploration model
        self.random_fuc = False
        self.dead_end_fuc = False
        self.counter_fuc = False
        
        if self.exploration_model == 'random':
            self.random_fuc = True
        elif self.exploration_model == 'deadend':
            self.dead_end_fuc = True
        elif self.exploration_model == 'counter':
            self.counter_fuc = True
        elif self.exploration_model == 'heuristic':
            self.dead_end_fuc = True
            self.counter_fuc = True
        
        # Initialize maps based on exploration model
        self.map_count = [[0 for _ in range(maze_dim)] for _ in range(maze_dim)]
        self.map_heuristic = [[min(abs(row - maze_dim // 2 + 1), abs(row - maze_dim // 2)) +
                               min(abs(col - maze_dim // 2 + 1), abs(col - maze_dim // 2))
                               for row in range(maze_dim)] for col in range(maze_dim)]

        # Example initialization, adjust as needed
        self.map_count[0][0] += 1
    def is_within_goal_bounds(self):
        x, y = self.location
        return x in self.goal_bounds and y in self.goal_bounds

    def check_backward(self, sensors):
        # Condition 1: First time encountering a dead zone (all sensors are 0)
        if not self.last_move_backward and all(sensor == 0 for sensor in sensors):
            return True

        # Condition 2: Last move was backward and both left and right sensors are 0
        if self.last_move_backward and sensors[0] == 0 and sensors[2] == 0:
            return True

        # Condition 3: Hit the goal, remember goal position and move backward to get out
        if self.in_goal_bounds():
            self.goal_position = self.location.copy()
            self.remember_goal = True
            return True

        return False
    
    def map(self, sensors):
        # Update map_maze for sensor readings
        for i, sensor in enumerate(sensors):
            if sensor > 0:
                direction_index = (i + self.map_dic[self.heading]) % 4
                self.map_maze[self.location[0]][self.location[1]][direction_index] += 1

        # Calculate reverse_index outside the condition block to ensure it's always defined
        reverse_index = (3 + self.map_dic[self.heading]) % 4

        # Update map_maze for the movement direction
        if not self.last_move_backward:
            self.map_maze[self.location[0]][self.location[1]][reverse_index] += 1
        else:
            reverse_location_x = self.location[0] + self.dir_move[self.heading][0]
            reverse_location_y = self.location[1] + self.dir_move[self.heading][1]
            self.map_maze[reverse_location_x][reverse_location_y][reverse_index] += 1

        # Update map_maze for previous locations in case of multiple movements
        if self.last_movement > 1:
            for n in range(1, self.last_movement):
                past_location_x = self.location[0] - n * self.dir_move[self.heading][0]
                past_location_y = self.location[1] - n * self.dir_move[self.heading][1]
                self.map_maze[past_location_x][past_location_y][reverse_index] += 1
                forward_index = (1 + self.map_dic[self.heading]) % 4
                self.map_maze[past_location_x][past_location_y][forward_index] += 1

    def calculate_coverage(self):
        uncovered_count = sum(row.count(0) for row in self.map_count)
        return 1 - float(uncovered_count) / self.maze_area

    def refresh_position(self, rotation, movement):
        # Determine if the movement is forward or backward
        is_forward_movement = movement > 0
        self.last_move_backward = not is_forward_movement

        # Update the heading based on rotation if moving forward
        if is_forward_movement:
            if rotation == -90:
                self.heading = self.dir_sensors[self.heading][0]
            elif rotation == 90:
                self.heading = self.dir_sensors[self.heading][2]

            # Update the location based on the heading and movement
            self.location[0] += self.dir_move[self.heading][0] * movement
            self.location[1] += self.dir_move[self.heading][1] * movement
        else:
            # If moving backward, record the current location and update the location
            self.map_dead_zone[self.location[0]][self.location[1]] = self.heading
            reverse_heading = self.dir_reverse[self.heading]
            self.location[0] += self.dir_move[reverse_heading][0]
            self.location[1] += self.dir_move[reverse_heading][1]

        # Update the map count at the new location
        self.map_count[self.location[0]][self.location[1]] += 1


    def check_dead_zone(self, sensors):
        if self.is_within_goal_bounds():
            self.goal_position = self.location[:]  # Update goal_position using list slicing
            self.remember_goal = True
            self.map_dead_zone[self.location[0]][self.location[1]] = self.heading
            return [[0, -1]]  # Return list with single element

        possible_moves = []
        for i, sensor_value in enumerate(sensors):
            if sensor_value > 0:
                rotation = self.base_rotation[i]
                max_move = min(sensor_value, 3)
                for movement in range(1, max_move + 1):
                    if rotation == -90:
                        check_heading = self.dir_sensors[self.heading][0]
                    elif rotation == 90:
                        check_heading = self.dir_sensors[self.heading][2]
                    else:
                        check_heading = self.dir_sensors[self.heading][1]

                    check_location_x = self.location[0] + self.dir_move[check_heading][0] * movement
                    check_location_y = self.location[1] + self.dir_move[check_heading][1] * movement

                    if self.map_dead_zone[check_location_x][check_location_y] == ' ':
                        possible_moves.append([rotation, movement])

        return possible_moves


    def update_and_export_maze(self):
        updated_map_maze = [[[0 for _ in range(4)] for _ in range(self.maze_dim)] for _ in range(self.maze_dim)]
        mapped_maze = [[0 for _ in range(self.maze_dim)] for _ in range(self.maze_dim)]

        # Update map_maze based on neighboring cells
        for x in range(self.maze_dim):
            for y in range(self.maze_dim):
                if y - 1 >= 0 and self.map_maze[x][y - 1][0] > 0:
                    updated_map_maze[x][y][2] += 1
                if x + 1 < self.maze_dim and self.map_maze[x + 1][y][3] > 0:
                    updated_map_maze[x][y][1] += 1
                if y + 1 < self.maze_dim and self.map_maze[x][y + 1][2] > 0:
                    updated_map_maze[x][y][0] += 1
                if x - 1 >= 0 and self.map_maze[x - 1][y][1] > 0:
                    updated_map_maze[x][y][3] += 1

        # Convert updated_map_maze to binary values and calculate mapped_maze
        for x in range(self.maze_dim):
            for y in range(self.maze_dim):
                for i in range(4):
                    updated_map_maze[x][y][i] = 1 if updated_map_maze[x][y][i] > 0 else 0
                mapped_maze[x][y] = updated_map_maze[x][y][0] + 2 * updated_map_maze[x][y][1]+ 4 * updated_map_maze[x][y][2] + 8 * updated_map_maze[x][y][3]

        # Export mapped_maze to a text file
        with open('maze.txt', 'w') as f:
            f.write(str(self.maze_dim) + '\n')
            for x in range(self.maze_dim):
                for y in range(self.maze_dim):
                    if y == self.maze_dim - 1:
                        f.write(str(mapped_maze[x][y]) + '\n')
                    else:
                        f.write(str(mapped_maze[x][y]) + ',')

        # Optionally, update instance variables if needed
        self.map_maze = updated_map_maze
        self.mapped_maze = mapped_maze
        
    def update_values(self):
        change = True
        while change:
            change = False
            for x in range(self.maze_dim):
                for y in range(self.maze_dim):
                    if self.goal_position[0] == x and self.goal_position[1] == y:
                        if self.value[x][y] > 0:
                            self.value[x][y] = 0
                            change = True
                    elif sum(self.map_maze[x][y]) > 0:
                        V2 = []
                        for a in range(4):
                            if self.map_maze[x][y][a] > 0:
                                if 0 <= x - (a - 2) * (a % 2)<self.maze_dim and 0 <= y - (a - 1) * ((a + 1) % 2)<self.maze_dim:
                                    V2.append(self.value[x - (a - 2) * (a % 2)][y - (a - 1) * ((a + 1) % 2)])
                        V2 = min(V2) + 1  # if len()
                        if V2 < self.value[x][y]:
                            change = True
                            self.value[x][y] = V2


    def next_move(self, sensors):
        if self.run == 0:
            self.map(sensors)
            if self.move_time > 999:
                print(self.move_time)
                print(self.calculate_coverage())
                self.show('map_count')
                self.rectify_maze()

            if self.remember_goal and (self.calculate_coverage() > 0.5 or self.move_time > 900):
                print('1st run moves :' + str(self.move_time))
                self.run = 1
                self.update_and_export_maze()
                self.update_values()
                self.location = [0, 0]
                self.heading = 'up'
                self.move_time = 0
                return 'Reset', 'Reset'

            self.move_time += 1

            if self.random_fuc:
                if self.check_backward(sensors):
                    rotation, movement = 0, -1
                    self.refresh_position(rotation, movement)
                    self.last_movement = movement
                    return rotation, movement

                valid_rotation = [self.base_rotation[i] for i in range(3) if sensors[i] > 0]
                if self.last_move_backward:
                    valid_rotation = [self.base_rotation[i] for i in [0, 2] if sensors[i] > 0]
                    self.last_move_backward = False

                if not valid_rotation:
                    rotation, movement = 0, -1
                    self.refresh_position(rotation, movement)
                    self.last_movement = movement
                    return rotation, movement

                rotation = random.choice(valid_rotation)
                if rotation == -90:
                    valid_movement = range(1, min(3, sensors[0]) + 1)
                elif rotation == 90:
                    valid_movement = range(1, min(3, sensors[2]) + 1)
                else:
                    valid_movement = range(1, min(3, sensors[1]) + 1)

                valid_movement = list(valid_movement)  # Convert to list to check its length
                if not valid_movement:
                    rotation, movement = 0, -1
                    self.refresh_position(rotation, movement)
                    self.last_movement = movement
                    return rotation, movement

                movement = random.choice(valid_movement)
                self.refresh_position(rotation, movement)
                self.last_movement = movement
                return rotation, movement

            if self.dead_end_fuc:
                valid_move = self.check_dead_zone(sensors)
                if not valid_move:
                    rotation, movement = 0, -1
                    self.refresh_position(rotation, movement)
                    self.last_movement = movement
                    return rotation, movement

                if self.random_fuc:
                    rotation, movement = random.choice(valid_move)
                    self.refresh_position(rotation, movement)
                    self.last_movement = movement
                    return rotation, movement

                if self.counter_fuc:
                    valid_position_count = []
                    for p in valid_move:
                        heading = self.dir_sensors[self.heading][int(p[0] / 90) + 1]
                        x = self.location[0] + self.dir_move[heading][0] * p[1]
                        y = self.location[1] + self.dir_move[heading][1] * p[1]
                        valid_position_count.append([self.map_count[x][y], self.map_heuristic[x][y], p])

                    valid_position_count.sort(key=operator.itemgetter(0, 1))
                    min_c = valid_position_count[0][0]
                    min_h = valid_position_count[0][1]
                    min_c_move = [move for move in valid_position_count if move[0] == min_c]

                    if len(min_c_move) == 1:
                        best_move = min_c_move[0]
                    elif self.exploration_model != 'heuristic':
                        best_move = random.choice(min_c_move)
                    else:
                        min_h_move = [move for move in min_c_move if move[1] == min_h]
                        if len(min_h_move) == 1:
                            best_move = min_h_move[0]
                        else:
                            best_move = next((move for move in min_h_move if move[0] == 0), None)
                            if not best_move:
                                best_move = random.choice(min_h_move)

                    rotation, movement = best_move[2]
                    self.refresh_position(rotation, movement)
                    self.last_movement = movement
                    return rotation, movement

        possible_rotation = [self.base_rotation[i] for i in range(3) if sensors[i] > 0]
        possible_move = [min(sensors[i], 3) for i in range(3) if sensors[i] > 0]

        if not possible_rotation or not possible_move:
            rotation, movement = 0, -1
            self.refresh_position(rotation, movement)
            self.last_movement = movement
            return rotation, movement

        best_move = [0, 0]
        min_value = self.value[self.location[0]][self.location[1]]
        for i in range(len(possible_rotation)):
            for j in range(1, possible_move[i] + 1):
                rotation = possible_rotation[i]
                movement = j

                if rotation == -90:
                    check_heading = self.dir_sensors[self.heading][0]
                elif rotation == 90:
                    check_heading = self.dir_sensors[self.heading][2]
                else:
                    check_heading = self.dir_sensors[self.heading][1]

                check_location_x = self.location[0] + self.dir_move[check_heading][0] * movement
                check_location_y = self.location[1] + self.dir_move[check_heading][1] * movement

                if self.value[check_location_x][check_location_y] < min_value:
                    min_value = self.value[check_location_x][check_location_y]
                    best_move = [rotation, movement]

        rotation, movement = best_move
        self.path_length += movement

        if rotation == -90:
            self.heading = self.dir_sensors[self.heading][0]
        elif rotation == 90:
            self.heading = self.dir_sensors[self.heading][2]
        else:
            pass

        self.location[0] += self.dir_move[self.heading][0] * movement
        self.location[1] += self.dir_move[self.heading][1] * movement
        self.move_time += 1

        if self.is_within_goal_bounds():
            self.map_location[0][0] = 'STRT'
            self.map_location[self.location[0]][self.location[1]] = 'FNSH'
            print('2nd run moves :' + str(self.move_time))
            print('2nd run path length :' + str(self.path_length))

        return rotation, movement
