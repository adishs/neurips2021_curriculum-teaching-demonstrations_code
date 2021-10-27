import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os


class navigation_dataset(torch.utils.data.Dataset):

    def __init__(self, data_path, difficulty="uniform", b=1, a=0, max_iters=None):
        self.X = []
        self.y = []
        self.locations = []
        self.file_name = []
        self.stats = []
        path = os.path.join(data_path, "np_arrays/")
        max_horizon = 0
        longest_index = -1
        self.min_reward = np.inf
        avg_optimal_reward = 0
        for f in os.listdir(path):
            arr = np.load(os.path.join(path, f))
            self.X.append(arr["input"])
            self.y.append(arr["label"])
            if len(self.y[-1]) > max_horizon:
                max_horizon = len(self.y[-1])
                longest_index = len(self.y) - 1
            self.locations.append(arr["initial_location"])
            self.file_name.append(os.path.join(path, f))
            stat_arr = list()
            stat_arr.append(arr["reward"])
            self.min_reward = min(self.min_reward, arr["reward"])
            avg_optimal_reward += arr["reward"]
            stat_arr.append(arr["turns"])
            stat_arr.append(arr["muds"])
            stat_arr.append(arr["goals"])
            stat_arr.append(arr["bombs"])
            stat_arr.append(arr["reachable"])
            stat_arr.append(arr["num_optimal_paths"])
            self.stats.append(stat_arr)

        assert len(self.X)==len(self.stats), "Data loading error."
        self.ordering = np.arange(len(self.X))
        self.iteration = 1
        self.b = b
        self.a = a
        self.max_iters = max_iters
        print ("Average optimal reward on tasks = {}".format(avg_optimal_reward/len(self.X)))
        #define teaching difficulty parameters.
        if difficulty == "uniform":
            self.difficulty = self.uniform_difficulty
        elif difficulty == "curr":
            self.difficulty = self.teacher_difficulty
        self.difficulty_array = self.compute_difficulty()


    def __len__(self):
        return len(self.ordering)


    def dimension(self):
        return self.X[0].shape[2]


    def curriculum_order(self, learner_diff):
        objective = np.array([])
        for i in range(len(self.X)):
            objective = np.append(objective, self.difficulty_array[i] - learner_diff(torch.Tensor(self.X[i]).permute(2, 0, 1), torch.LongTensor(self.locations[i]), torch.LongTensor(self.y[i])))
        self.ordering = np.argsort(objective)
        self.ordering = self.ordering[:self.randomized_curriculum()]
        self.ordering = np.random.permutation(self.ordering)
        self.iteration += 1
        return


    def compute_difficulty(self):
        diff_array = np.array([])
        for i in range(len(self.X)):
            diff_array = np.append(diff_array, self.difficulty(i))
        if not np.all(diff_array == 1.): diff_array = (diff_array - np.min(diff_array)) / (np.max(diff_array) - np.min(diff_array))
        return np.log(diff_array + 1e-3)


    def uniform_difficulty(self, index):
        return 1.


    def teacher_difficulty(self, index):
        #goals * num_optimal_paths / reward
        return self.stats[index][3] * self.stats[index][6] / (self.stats[index][0] - self.min_reward + 1)


    def randomized_curriculum(self, pacing_fn="linear"):
        if pacing_fn == "linear":
            limit = self.linear_pacing()
        return int(limit)


    def linear_pacing(self):
        if self.a == 0:
            return self.b*len(self.X)

        return self.b*len(self.X) + min((self.iteration/(self.a*self.max_iters)), 1) * (1-self.b) * len(self.X)


    def __getitem__(self, i):
        return torch.Tensor(self.X[self.ordering[i]]).permute(2, 0, 1), torch.LongTensor(self.y[self.ordering[i]]), torch.LongTensor(self.locations[self.ordering[i]]), self.file_name[self.ordering[i]]


def pad_label_collate(batch):
    (input, labels, locations, file_names) = zip(*batch)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)

    return torch.stack(input), padded_labels, torch.stack(locations), list(file_names)
