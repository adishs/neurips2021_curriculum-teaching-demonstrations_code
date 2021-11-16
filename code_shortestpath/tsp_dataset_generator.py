import numpy as np
import copy
import os
import json
import heapq
import itertools


class tsp_data_generator:

    def __init__(self, goals, muds, data_dir):
        self.size = 6
        self.goals = goals
        self.muds = muds
        self.data_dir = data_dir
        if not os.path.isdir(data_dir):
            os.makedirs(os.path.join(data_dir, "files"))
            os.makedirs(os.path.join(data_dir, "programs"))
            os.makedirs(os.path.join(data_dir, "np_arrays"))

        self.directions = {0:"east", 1: "south", 2:"west", 3:"north"}
        self.direction_step = np.array([[0,1,0],[1,0,0],[0,-1,0],[-1,0,0]])
        self.codes = {0:"move", 1:"turn_left", 2:"turn_right", 3:"collect_key"}


    def sample_task(self):
        self.grid = np.zeros((self.size, self.size))
        grid_greedy = np.zeros((self.size, self.size))
        sampled_locations = list(np.random.choice(np.arange(self.size**2), 1+self.goals+self.muds, replace=False))
        initial_location = np.append(self.convert_to_2d(sampled_locations.pop()), np.random.randint(4))
        self.grid[tuple(initial_location[:2])] = 1
        grid_greedy[tuple(initial_location[:2])] = 1
        goal_locations = list()
        for _ in range(self.goals):
            goal_locations.append(self.convert_to_2d(sampled_locations.pop()))
            self.grid[tuple(goal_locations[-1])] = 2
            grid_greedy[tuple(goal_locations[-1])] = 2
        
        for _ in range(self.muds):
            loc = self.convert_to_2d(sampled_locations.pop())
            self.grid[tuple(loc)] = 3
            grid_greedy[tuple(loc)] = 3
        
        assert len(sampled_locations)==0, "Incorrect locations sampled."

        tour_rewards = list()
        #iterate over possible tours.
        for p in itertools.permutations(np.arange(self.goals)):
            current_location = copy.deepcopy(initial_location)
            current_reward = 0
            current_tour = []
            total_turns = 0
            for i in p:
                reward, path = self.find_max_reward_paths(current_location, goal_locations[i])
                if reward != -np.inf:
                    path = path[np.random.randint(len(path))]
                    current_location = path[0]
                    current_tour += path[1]
                    total_turns += path[2]
                current_reward += reward
            reward, path = self.find_max_reward_paths(current_location, initial_location[:2])
            if reward != -np.inf:
                path = path[np.random.randint(len(path))]
                current_tour += path[1]
                total_turns += path[2]
            current_reward += reward

            tour_list = [current_reward, current_tour, total_turns]
            tour_rewards.append(copy.deepcopy(tour_list))
        tour_rewards.sort(key=lambda l:l[0], reverse=True)
        if tour_rewards[0][0] == tour_rewards[1][0]:
            return None, None, None

        #greedy max_reward_paths.
        greedy_reward = 0
        greedy_path = []
        greedy_turns = 0
        current_location = copy.deepcopy(initial_location)
        for _ in range(self.goals):
            reward, path = self.find_shortest_path(current_location, grid_greedy)
            current_location = path[0]
            grid_greedy[tuple(current_location[:2])] = 1
            greedy_reward += reward
            greedy_path += path[1]
            greedy_turns += path[2]
        grid_greedy[tuple(initial_location[:2])] = 2
        reward, path = self.find_shortest_path(current_location, grid_greedy)
        greedy_reward += reward
        greedy_path += path[1]
        greedy_turns += path[2]

        return tour_rewards[0], [greedy_reward, greedy_path, greedy_turns], initial_location


    def find_max_reward_paths(self, initial_location, goal_location):
        weights = np.ones((self.size, self.size, 4)) * np.inf
        weights[tuple(initial_location)] = 0
        max_reward = -np.inf
        max_reward_paths = []
        count = 0

        heap = [(0, copy.deepcopy(count), [initial_location, [], 0])]
        heapq.heapify(heap)

        while (len(heap) > 0):
            cost, _, node_original = heapq.heappop(heap)
            for m in range(3):
                node = copy.deepcopy(node_original)
                location = node[0]
                new_cost = cost + 1
                path = copy.deepcopy(node[1])
                path.append(self.codes[m])
                turns = node[2]
                if m == 0:
                    location = location + self.direction_step[location[2]]
                elif m == 1:
                    location[2] = (location[2]-1)%4
                    turns += 1
                elif m == 2:
                    location[2] = (location[2]+1)%4
                    turns += 1

                if self.is_valid(location):
                    if np.all(location[:2] == goal_location):
                        reward = -new_cost
                        if reward == max_reward:
                            max_reward_paths.append([location, path, turns])
                        elif reward > max_reward:
                            max_reward = reward
                            max_reward_paths.clear()
                            max_reward_paths.append([location, path, turns])
                        continue
                    elif self.grid[tuple(location[:2])] == 3:
                        new_cost += 1
                    elif not np.all(location[:2]==initial_location[:2]) and (self.grid[tuple(location[:2])] == 1 or self.grid[tuple(location[:2])] == 2):
                        continue

                    if weights[tuple(location)] > new_cost:
                        weights[tuple(location)] = new_cost
                        count += 1
                        new_node = (new_cost, copy.deepcopy(count), [location, path, turns])
                        heapq.heappush(heap, new_node)

        return max_reward, max_reward_paths


    def find_shortest_path(self, initial_location, grid):
        weights = np.ones((self.size, self.size, 4)) * np.inf
        weights[tuple(initial_location)] = 0
        max_reward = -np.inf
        max_reward_path = []
        count = 0

        heap = [(0, copy.deepcopy(count), [initial_location, [], 0])]
        heapq.heapify(heap)

        while (len(heap) > 0):
            cost, _, node_original = heapq.heappop(heap)

            for m in range(3):
                node = copy.deepcopy(node_original)
                location = node[0]
                new_cost = cost + 1
                path = copy.deepcopy(node[1])
                path.append(self.codes[m])
                turns = node[2]
                if m == 0:
                    location = location + self.direction_step[location[2]]
                elif m == 1:
                    location[2] = (location[2]-1)%4
                    turns += 1
                elif m == 2:
                    location[2] = (location[2]+1)%4
                    turns += 1

                if self.is_valid(location):
                    if grid[tuple(location[:2])] == 2:
                        reward = -new_cost
                        if reward > max_reward:
                            max_reward = reward
                            max_reward_path = [location, path, turns]
                        continue
                    elif grid[tuple(location[:2])] == 3:
                        new_cost += 1

                    if weights[tuple(location)] > new_cost:
                        weights[tuple(location)] = new_cost
                        count += 1
                        new_node = (new_cost, copy.deepcopy(count), [location, path, turns])
                        heapq.heappush(heap, new_node)

        return max_reward, max_reward_path


    def convert_to_2d(self, loc):
        return np.array([loc//self.size, loc%self.size])


    def is_valid(self, location):
        if location[0]>=0 and location[0]<self.size and location[1]>=0 and location[1]<self.size:
            return True
        return False


    def generate_data(self, data_size):
        count = 0
        while count < data_size:
            tour, greedy_tour, initial_location = self.sample_task()
            if tour is None:
                continue
            self.print_grid_and_program(self.grid, tour, initial_location, greedy_tour)
            self.save_np_arrays(self.grid, tour, initial_location, greedy_tour)
            count += 1


    def save_np_arrays(self, grid, tour, initial_location, greedy_tour):
        token_dict = {"move": 0, "turn_left": 1, "turn_right": 2}
        label_array = np.array([])
        for line in tour[1]:
            label_array = np.append(label_array, token_dict[line])

        assert len(label_array)==len(tour[1]), "error with label array!"

        # features - 4xdirections, initial_location, goals, muds
        input_array = np.zeros((self.size, self.size, 7))
        input_array[tuple(initial_location)] = 1
        for i in range(self.size):
            for j in range(self.size):
                if grid[i, j] == 1:
                    input_array[i, j, 4] = 1
                elif grid[i,j] == 2:
                    input_array[i, j, 5] = 1
                elif grid[i, j] == 3:
                    input_array[i, j, 6] = 1

        np.savez(os.path.join(self.data_dir, "np_arrays", "datapoint_{}".format(self.number)),\
            input=input_array, label=label_array, initial_location=initial_location,\
            reward=tour[0], greedy_reward=greedy_tour[0], turns=tour[2], goals=self.goals, muds=self.muds)
        return


    def print_grid_and_program(self, grid, tour, initial_location, greedy_tour):
        file_dir = os.path.join(self.data_dir, "files/")
        self.number = len(os.listdir(file_dir))
        with open(file_dir + "task_{}.txt".format(self.number), 'w') as f:
            f.write("gridsz\t({},{})\n".format(self.size, self.size))
            f.write("\n")
            f.write("pregrid")
            for i in range(1, self.size+1):
                f.write("\t{}".format(i))
            f.write("\n")
            for i in range(self.size):
                f.write("{}".format(i+1))
                for j in range(self.size):
                    f.write("\t")
                    if grid[i, j] == 1:
                        f.write("S")
                    elif grid[i, j] == 2:
                        f.write("G")
                    elif grid[i, j] == 3:
                        f.write("#")
                    else:
                        f.write(".")
                f.write("\n")
            f.write("agentloc\t({},{})\n".format(initial_location[0]+1, initial_location[1]+1))
            f.write("agentdir\t{}\n".format(self.directions[initial_location[2]]))


        json_dict = {"agent": tour[1], "greedy": greedy_tour[1]}
        code_dir = os.path.join(self.data_dir, "programs/")
        with open(code_dir + "program_{}.json".format(self.number), 'w') as f:
            json.dump(json_dict, f, indent=4)

        return


def main():
    dataset_dir = "./datasets/tsp_data/{}/"

    max_goals = 4
    mode = ["train", "test", "val"]
    data_size = [2000, 500, 100]

    print ("Begin data generation.")
    for i, m in enumerate(mode):
        data_path = dataset_dir.format(m)
        for goals in range(2, max_goals+1):
            generator = tsp_data_generator(goals, 0, data_path)
            generator.generate_data(data_size[i])
        print ("{} dataset generated.".format(m))

    return


if __name__=="__main__":
    main()
