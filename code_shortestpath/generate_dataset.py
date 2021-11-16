import numpy as np
import copy
import os
import json
import heapq


class data_generator:
    """
    @brief: Class to generate code, task pairs.
    """

    def __init__(self, goals, muds, bombs, data_dir):
        self.size = 6
        self.goals = goals
        self.muds = muds
        self.bombs = bombs
        self.data_dir = data_dir
        if not os.path.isdir(data_dir):
            os.makedirs(os.path.join(data_dir, "files"))
            os.makedirs(os.path.join(data_dir, "programs"))
            os.makedirs(os.path.join(data_dir, "np_arrays"))

        self.directions = {0:"east", 1: "south", 2:"west", 3:"north"}
        self.direction_step = np.array([[0,1,0],[1,0,0],[0,-1,0],[-1,0,0]])
        self.codes = {0:"move", 1:"turn_left", 2:"turn_right"}


    def sample_task(self):
        self.grid = np.zeros((self.size, self.size))
        sampled_locations = list(np.random.choice(np.arange(self.size**2), 1+self.goals+self.muds+self.bombs, replace=False))

        initial_location = np.append(self.convert_to_2d(sampled_locations.pop()), np.random.randint(4))
        goal_locations = list()
        for _ in range(self.goals):
            goal_locations.append(self.convert_to_2d(sampled_locations.pop()))
            self.grid[tuple(goal_locations[-1])] = 2 + len(goal_locations)

        for _ in range(self.muds):
            self.grid[tuple(self.convert_to_2d(sampled_locations.pop()))] = 1

        for _ in range(self.bombs):
            self.grid[tuple(self.convert_to_2d(sampled_locations.pop()))] = 2

        assert len(sampled_locations)==0, "Incorrect locations sampled."

        #find min cost (max reward) paths.
        max_reward = -np.inf
        rewards_list = []
        max_reward_paths = []
        optimal_path = None
        reachable = 0
        for goal_loc in goal_locations:
            reward, paths = self.find_max_reward_paths(initial_location, goal_loc)
            rewards_list.append(reward)
            max_reward_paths.append(paths)
            if reward > max_reward:
                max_reward = reward
                optimal_path = paths
            if reward > -np.inf:
                reachable += 1

        #Rejection criteria.
        if reachable == 0:
            return None, None, None, None, None
        elif max_reward >= 7:
            return None, None, None, None, None
        elif len(rewards_list)==2 and abs(rewards_list[0] - rewards_list[1]) <= 2:
            return None, None, None, None, None

        #saving only optimal paths
        return self.grid, max_reward, optimal_path, initial_location, reachable


    def convert_to_2d(self, loc):
        return np.array([loc//self.size, loc%self.size])


    def find_max_reward_paths(self, initial_location, goal_location):
        weights = np.ones((self.size, self.size, 4)) * np.inf
        weights[tuple(initial_location)] = 0
        max_reward = -np.inf
        max_reward_paths = []
        count = 0

        #(cost, location, path, turns, muds)
        heap = [(0, copy.deepcopy(count), [initial_location, [], 0, 0])]
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
                muds = node[3]
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
                        reward = 10 - new_cost
                        if reward == max_reward:
                            max_reward_paths.append([path, turns, muds])
                        elif reward > max_reward:
                            max_reward = reward
                            max_reward_paths.clear()
                            max_reward_paths.append([path, turns, muds])
                        continue
                    elif self.grid[tuple(location[:2])] >= 2:
                        continue
                    elif self.grid[tuple(location[:2])] == 1:
                        new_cost += 1
                        muds += 1

                    if weights[tuple(location)] > new_cost:
                        weights[tuple(location)] = new_cost
                        count += 1
                        new_node = (new_cost, copy.deepcopy(count), [location, path, turns, muds])
                        heapq.heappush(heap, new_node)

        return max_reward, max_reward_paths


    def is_valid(self, location):
        if location[0]>=0 and location[0]<self.size and location[1]>=0 and location[1]<self.size:
            return True
        return False


    def generate_data(self, data_size):
        count = 0
        while count < data_size:
            grid, max_reward, optimal_path, initial_location, reachable = self.sample_task()
            if grid is None:
                continue
            num_optimal_paths = len(optimal_path)
            for path in optimal_path[:3]:
                self.print_grid_and_program(grid, path, initial_location, reachable)
                self.save_np_arrays(grid, max_reward, path, initial_location, reachable, num_optimal_paths)
                count += 1

        return


    def save_np_arrays(self, grid, max_reward, path, initial_location, reachable, num_optimal_paths):
        token_dict = {"move": 0, "turn_left": 1, "turn_right": 2}
        label_array = np.array([])
        for line in path[0]:
            label_array = np.append(label_array, token_dict[line])

        assert len(label_array)==len(path[0]), "error with label array!"

        # features - 4xlocation, muds, bombs, goals
        input_array = np.zeros((self.size, self.size, 7))
        input_array[tuple(initial_location)] = 1
        for i in range(self.size):
            for j in range(self.size):
                if grid[i, j] == 1:
                    input_array[i, j, 4] = 1
                elif grid[i,j] == 2:
                    input_array[i, j, 5] = 1
                elif grid[i,j] > 2:
                    input_array[i, j, 6] = 1

        np.savez(os.path.join(self.data_dir, "np_arrays", "datapoint_{}".format(self.number)), \
            input=input_array, label=label_array, initial_location=initial_location, reward=max_reward,\
            turns=path[1], muds=path[2], goals=self.goals, bombs=self.bombs, reachable=reachable, num_optimal_paths=num_optimal_paths)
        return


    def print_grid_and_program(self, grid, path, initial_location, reachable):
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
                        f.write("#")
                    elif grid[i, j] == 2:
                        f.write("+")
                    elif grid[i, j] > 2:
                        f.write("G")
                    else:
                        f.write(".")
                f.write("\n")
            f.write("agentloc\t({},{})\n".format(initial_location[0]+1, initial_location[1]+1))
            f.write("agentdir\t{}\n".format(self.directions[initial_location[2]]))
            f.write("reachable\t{}".format(reachable))


        json_dict = {"agent": path[0]}
        code_dir = os.path.join(self.data_dir, "programs/")
        with open(code_dir + "program_{}.json".format(self.number), 'w') as f:
            json.dump(json_dict, f, indent=4)

        return


def main():
    dataset_dir = "./datasets/shortest_path_data/{}/"
    max_muds = max_bombs = 12
    max_goals = 2
    mode = ["train", "test", "val"]
    data_size = [50, 15, 5]
    
    print ("Begin data generation.")
    for i, m in enumerate(mode):
        data_path = dataset_dir.format(m)
        for g in range(1, max_goals+1):
            for muds in range(0, max_muds+1):
                for bombs in range(0, max_bombs+1):
                    generator = data_generator(g, muds, bombs, data_path)
                    generator.generate_data(data_size[i])
        print ("{} dataset generated.".format(m))

    print ("Data generation complete.")
    return


if __name__=="__main__":
    main()
