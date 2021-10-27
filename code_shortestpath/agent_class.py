import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from dataloader import *
from model_files.task_embedding_network import *
from model_files.policy_network import *
import copy
import os

class agent:

    def __init__(self, grid_size, action_space, learner_curriculum_type, teacher_curriculum_type, in_features, data_dir, lr, batch_size, max_epoch, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print (self.device)
        self.p = 0.1
        self.size = grid_size
        self.actions = action_space
        self.in_features = in_features
        self.data_dir = data_dir
        self.lr = lr
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.save_dir = None
        self.direction_step = torch.LongTensor([[0,1,0],[1,0,0],[0,-1,0],[-1,0,0]]).to(self.device)
        self.horizon_len = {6: 30}

        self.policy_net = self.instantiate_model()
        self.policy_net.to(self.device)
        print ("Neural network created.")

        self.learner_curriculum_type = learner_curriculum_type
        self.teacher_curriculum_type = teacher_curriculum_type
        self.func = None
        if learner_curriculum_type == "curr":
            self.func = self.learner_difficulty
        elif learner_curriculum_type == "uniform":
            self.func = self.uniform_difficulty

        if len(kwargs) > 0:
            self.train_set = navigation_dataset(data_dir + "train/", teacher_curriculum_type, kwargs["b"], kwargs["a"], self.max_epoch)
        else:
            self.train_set = navigation_dataset(data_dir + "train/", teacher_curriculum_type)
        self.val_set = navigation_dataset(data_dir + "val/")
        self.test_set = navigation_dataset(data_dir + "test/")
        print ("Dataset loaded.")

        self.step = 0
        self.train_performance_step = np.array([])
        self.val_performance_step = np.array([])
        self.test_performance_step = np.array([])
        self.train_performance = np.array([])
        self.val_performance = np.array([])
        self.test_performance = np.array([])
        self.curriculum = list()
        return


    def set_save_path(self, identifier):
        self.save_dir = os.path.join(self.data_dir, identifier)
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        return


    def instantiate_model(self):
        """
        @brief: Create policy neural network model.
        """
        task_encoder = task_embedding_net(self.in_features, self.size)
        policy_net = policy_network(task_encoder, self.actions)
        return policy_net


    def load_dataset(self):
        if self.func is None:
            train_data = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, collate_fn=pad_label_collate)
        else:
            train_data = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, collate_fn=pad_label_collate)
        return train_data


    def train_model(self):
        """
        @brief: Function that performs model training.
        """
        train_data = self.load_dataset()

        #Define training criteria
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = optim.SGD(self.policy_net.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.5)
        #save starting performance
        val_reward = self.test_model(self.val_set)
        best_val_reward = val_reward
        best_model = copy.deepcopy(self.policy_net)
        self.val_performance_step = np.append(self.val_performance_step, val_reward)
        test_reward = self.test_model(self.test_set)
        self.test_performance_step = np.append(self.test_performance_step, test_reward)

        print ("Begin training.")
        for epoch in range(self.max_epoch):
            epoch_loss = 0.
            epoch_correct = 0.
            epoch_linecount = 0.
            code_correct = 0.
            code_count = 0.

            #Order samples based on curriculum at the start of every epoch.
            if self.func is not None:
                self.train_set.curriculum_order(self.func)

            self.policy_net.train()
            for i, data in enumerate(train_data):
                self.step += 1
                task = data[0].to(self.device)
                labels = data[1].to(self.device)
                locations = data[2].to(self.device)
                self.curriculum.append(data[3])

                optimizer.zero_grad()
                loss = 0.

                flag = torch.ones(labels.shape[0], dtype=torch.bool, device=self.device)
                for l in range(labels.shape[1]):
                    outputs = self.policy_net(task)
                    _, predicted = torch.max(outputs, 1)
                    loss += criterion(outputs, labels[:, l])
                    #update task grid
                    _, task, locations = self.update_tasks(task, locations, labels[:, l])
                    epoch_correct += (predicted==labels[:, l]).sum().item()
                    epoch_linecount += labels.shape[0] - (labels[:, l]==-1).sum().item()
                    flag = torch.logical_and(flag, torch.logical_or(predicted==labels[:, l], labels[:, l]==-1) )

                code_correct += flag.sum().item()
                code_count += labels.shape[0]
                train_accuracy = np.array([100*epoch_correct/epoch_linecount, 100*code_correct/code_count])
                loss.backward()
                optimizer.step()
                if (i+1) % (len(self.train_set)//self.batch_size//4) == 0:
                    state = "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}, Full Code: {:.2f}".format(epoch+1, self.max_epoch, i+1, len(self.train_set)//self.batch_size, loss.item(), train_accuracy[0], train_accuracy[1])
                    print (state)

                if self.step%100 == 0:
                    scheduler.step()
                    self.train_performance_step = np.append(self.train_performance_step, train_accuracy[1])
                    #keep track of best performing model on val set.
                    val_reward = self.test_model(self.val_set)
                    self.val_performance_step = np.append(self.val_performance_step, val_reward)
                    if val_reward >= best_val_reward:
                        best_model = copy.deepcopy(self.policy_net)
                        best_val_reward = val_reward

                    test_reward = self.test_model(self.test_set, best_model)
                    self.test_performance_step = np.append(self.test_performance_step, test_reward)
                    self.policy_net.train()

            #Run validation
            val_reward = self.test_model(self.val_set)

            #Save best model for early stopping
            if val_reward >= best_val_reward:
                best_model = copy.deepcopy(self.policy_net)
                if self.save_dir is not None:
                    torch.save(self.policy_net.state_dict(), os.path.join(self.save_dir, "model_lr={}_batch={}_task={}.pt".format(self.lr, self.batch_size, self.policy_net.task_embedding_net.embedding_size)))
                best_val_reward = val_reward
            state = "Validation reward = {:.2f}".format(val_reward)
            print (state)

            self.train_performance = np.append(self.train_performance, train_accuracy[1])
            self.val_performance = np.append(self.val_performance, val_reward)
            test_reward = self.test_model(self.test_set, best_model)
            self.test_performance = np.append(self.test_performance, test_reward)

        return


    def update_tasks(self, task_o, location_o, tokens):
        reward = []
        task = task_o.detach().clone()
        location = location_o.detach().clone()
        for i, t in enumerate(tokens):
            if t==-1: continue
            cost = 1
            if torch.all(location[i] == -1):
                print ("Entered ended task.")
                print (tokens[i])
                exit(0)
                continue

            task[i, location[i][2]][tuple(location[i][:2])] = 0
            if np.random.rand() < self.p:
                t = np.random.choice(np.arange(self.actions))

            if t == 0:
                location[i] = location[i] + self.direction_step[location[i][2]]
            elif t == 1:
                location[i][2] = (location[i][2]-1)%4
            elif t == 2:
                location[i][2] = (location[i][2]+1)%4

            if self.is_valid(location[i]):
                task[i, location[i][2]][tuple(location[i][:2])] = 1
                if task[i, 4][tuple(location[i][:2])] == 1: #mud
                    cost += 1
                elif task[i, 5][tuple(location[i][:2])] == 1: #bomb
                    cost += 5
                    location[i] = -torch.ones(location[i].shape)
                elif task[i, 6][tuple(location[i][:2])] == 1: #goal
                    cost -= 10
                    location[i] = -torch.ones(location[i].shape)
            else:
                cost += 5
                location[i] = -torch.ones(location[i].shape)
            reward.append(-cost)
        return reward, task, location


    def is_valid(self, location):
        if location[0]>=0 and location[0]<self.size and location[1]>=0 and location[1]<self.size:
            return True
        return False


    def test_model(self, dataset=None, net=None):
        if dataset is None:
            dataset = self.test_set
        test_data = DataLoader(dataset, batch_size=1, shuffle=False)

        if net is None:
            net = self.policy_net

        horizon_exceeded = 0
        net.eval()
        with torch.no_grad():
            total_reward = 0.

            # for task, labels, locations, filenames in test_data:
            for data in test_data:
                task = data[0].to(self.device)
                labels = data[1].to(self.device)
                locations = data[2].to(self.device)

                steps = 0
                while torch.any(locations[0] != -1) and steps < self.horizon_len[self.size]:
                    outputs = net(task)
                    _, predicted = torch.max(outputs, 1)
                    reward, task, locations = self.update_tasks(task, locations, predicted)
                    total_reward += reward[0]
                    steps += 1
        return total_reward/len(test_data)


    def learner_difficulty(self, task, location, labels):
        return -(self.policy_log_likelihood(task.to(self.device), location.to(self.device), labels.to(self.device)))


    def policy_log_likelihood(self, task, location, labels):
        self.policy_net.eval()
        log_likelihood = 0.
        prob_layer = torch.nn.LogSoftmax(dim=1)
        task = task.unsqueeze(0)
        location = location.unsqueeze(0)
        labels = labels.unsqueeze(0)
        with torch.no_grad():
            task = task.to(self.device)
            for l in range(labels.shape[1]):
                outputs = self.policy_net(task)
                outputs = prob_layer(outputs)
                log_likelihood += outputs[0, labels[:, l]]
                _, task, location = self.update_tasks(task, location, labels[:, l])

        return log_likelihood.cpu()


    def uniform_difficulty(self, task=None, location=None, labels=None, key=None):
        return 1.
