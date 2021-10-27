import numpy as np
import copy
import collections
import sys
# sys.path.append("../IRL/")
import mdp_solver_class as mdp_solver


class policy_gradient_teacher:

    def __init__(self, env, eta, max_iters, len_episode, train_iterations, num_trajectories, project, teacher_type, D_init=None):
        self.env = env
        self.eta = eta
        self.max_iters = max_iters
        self.len_episode = len_episode
        self.train_iterations = train_iterations
        self.num_trajectories = num_trajectories
        self.project = project
        if D_init is None:
            self.D_init = self.env.D_init

        self.b = 0.
        self.a = 0.8

        self.radius = 10 * np.sqrt(self.env.n_features * 2)
        self.teacher_type = teacher_type
        if teacher_type == "pg":
            self.theta = np.ones(self.env.n_features * 2)
            self.pi = self.compute_policy()
            self.train()
        else:
            self.expert = mdp_solver.solver(env, env.true_reward, "deterministic")
            self.pi = self.expert.policy
            self.exp_reward = self.compute_exp_reward()
            self.lane_rewards = self.per_lane_reward()

        self.demonstrations = self.collect_trajectories()
        ## Teacher-Cur ##
        self.seen_array = np.zeros(len(self.env.initial_states)*self.num_trajectories)
        self.seen_states = np.zeros(len(self.env.initial_states))


    def compute_policy(self, env=None):
        if env is None:
            env = self.env

        pi = np.zeros((env.n_states, env.n_actions))
        for s in range(pi.shape[0]):
            pi[s] = self.action_distribution(s, env)
        return pi


    def action_distribution(self, state, env, theta=None):
        if theta is None:
            theta = self.theta

        action_dist = np.zeros(env.n_actions)
        for a in range(env.n_actions):
            action_dist[a] = np.exp(np.dot(theta[:self.env.n_features], env.action_features[a, state]) + np.dot(theta[self.env.n_features:], env.action_features[a, state])**2 )
        action_dist = action_dist / np.sum(action_dist)
        return action_dist


    def train(self):
        print ("Teacher's initial expected reward = {0:0.2f}".format(self.compute_exp_reward()))

        for iteration in range(self.train_iterations):
            if (iteration+1)%100 == 0:
                print ("Iteration [{}/{}]".format(iteration+1, self.train_iterations))
            
            for state in self.env.initial_states:
                episode = list()
                G = 0
                old_theta = copy.deepcopy(self.theta)

                for step in range(self.len_episode):
                    action = self.sample_action(state)
                    next_state = self.sample_state(state, action)
                    reward = self.env.true_reward[state]

                    episode.append([state, action, reward])
                    state = next_state

                #unroll episode and update theta
                for t in reversed(range(len(episode))):
                    state, action, reward = episode[t]
                    G += reward
                    self.gradient_update(G, state, action, old_theta)

                #update teacher's policy based on updated theta
                self.pi = self.compute_policy()

        self.exp_reward = self.compute_exp_reward()
        self.lane_rewards = self.per_lane_reward()
        print ("Teacher's expected reward after training = {0:0.2f}".format(self.exp_reward))
        return


    def sample_action(self, state):
        return np.random.choice(np.arange(self.env.n_actions), p=self.pi[state])


    def sample_state(self, state, action):
        return np.random.choice(np.arange(self.env.n_states), p=self.env.T_dense[action, state, :])


    def gradient_update(self, G, state, action, old_theta):
        self.theta += self.eta * G * np.append(self.env.action_features[action, state], 2*np.dot(old_theta[self.env.n_features:], self.env.action_features[action, state])*self.env.action_features[action, state])
        dist = self.action_distribution(state, self.env, old_theta)
        for a in range(self.env.n_actions):
            self.theta -= self.eta * G * dist[a] * np.append(self.env.action_features[a, state], 2*np.dot(old_theta[self.env.n_features:], self.env.action_features[a, state])*self.env.action_features[a, state])
        
        if self.project: self.project_theta()
        return


    def project_theta(self):
        if np.linalg.norm(self.theta) > self.radius:
            self.theta = self.theta * self.radius / np.linalg.norm(self.theta)


    def compute_exp_reward(self, D_init=None, env=None):
        if env is None:
            env = self.env
        
        if self.teacher_type == "vi":
            if env is None:
                expert = self.expert
            else:
                expert = mdp_solver.solver(env, env.true_reward, "deterministic")

        if D_init is None:
            D_init = env.D_init

        if self.teacher_type == "vi":
            rho_teacher = expert.compute_exp_rho_bellman()
        else:
            rho_teacher = env.compute_exp_rho_bellman(self.compute_policy(env), D_init)
        return env.reward_for_rho(rho_teacher)


    def per_lane_reward(self, env=None):
        if env is None:
            env = self.env

        reward_list = np.zeros(env.lanes)
        for l in range(env.lanes):
            D_init = np.zeros(env.n_states)
            for i in range(env.n_lanes):
                D_init[(l + i*env.lanes)*2*env.road_length] = 1/env.n_lanes
            reward_list[l] = self.compute_exp_reward(D_init, env)

        return reward_list


    def collect_trajectories(self):
        demonstrations = collections.defaultdict(list)

        for s in self.env.initial_states:
            for _ in range(self.num_trajectories):
                trajectory = self.sample_trajectory(self.len_episode, s)
                demonstrations[s].append(trajectory)

        return demonstrations


    def sample_trajectory(self, length, state=None):
        if state is None:
            state = np.random.choice(self.env.initial_states, 1)[0]

        trajectory = list()
        for t in range(length):
            action = self.sample_action(state)
            trajectory.append([state, action])

            state = self.sample_state(state, action)

        return trajectory


    def random_teaching(self):
        random_state = np.random.choice(np.arange(self.env.n_states), p=self.D_init)
        return [self.sample_trajectory(self.len_episode, random_state)], random_state


    def random_state_teaching(self):
        random_state = np.random.choice(np.arange(self.env.n_states), p=self.D_init)
        return self.demonstrations[random_state], random_state


    def importance_sampling_score(self, trajectory, learner):
        log_ratio = 0.
        for s,a in trajectory:
            log_ratio += np.log(self.pi[s,a]) - np.log(learner.pi[s,a])
        return log_ratio


    def teacher_importance_score(self, trajectory):
        log_ratio = 0.
        for s,a in trajectory:
            log_ratio += np.log(self.pi[s,a])
        return log_ratio


    def learner_importance_score(self, trajectory, learner):
        log_ratio = 0.
        for s,a in trajectory:
            log_ratio -= np.log(learner.pi[s,a])
        return log_ratio


    def curriculum_teaching(self, learner):
        traj_list = list()

        for s, trajectories in self.demonstrations.items():
            for trajectory in trajectories:
                traj_list.append([self.importance_sampling_score(trajectory, learner), trajectory, s])

        traj_list.sort(key=lambda l:l[0], reverse=True)
        return [traj_list[0][1]], traj_list[0][2]


    def teacher_curr_teaching(self):
        traj_list = list()
        if np.sum(self.seen_array) == len(self.env.initial_states)*self.num_trajectories:
            self.seen_array[:] = 0

        for s, trajectories in self.demonstrations.items():
            for i, trajectory in enumerate(trajectories):
                index = (s//(2*self.env.road_length))*self.num_trajectories + i
                if self.seen_array[index] == 1:
                    continue
                traj_list.append([self.teacher_importance_score(trajectory), trajectory, s, index])

        traj_list.sort(key=lambda l:l[0], reverse=True)
        self.seen_array[traj_list[0][3]] = 1
        return [traj_list[0][1]], traj_list[0][2]


    def learner_curr_teaching(self, learner):
        traj_list = list()

        for s, trajectories in self.demonstrations.items():
            for trajectory in trajectories:
                traj_list.append([self.learner_importance_score(trajectory, learner), trajectory, s])

        traj_list.sort(key=lambda l:l[0], reverse=True)
        return [traj_list[0][1]], traj_list[0][2]


    def curriculum_state_teaching(self, learner, iteration=-1):
        state_list = list()

        for s, trajectories in self.demonstrations.items():
            cost = 0
            for trajectory in trajectories:
                cost += self.importance_sampling_score(trajectory, learner)
            
            state_list.append([cost, s])

        state_list.sort(key=lambda l:l[0], reverse=True)
        index = self.randomizer(iteration)
        opt_state = state_list[index][1] #randomization.
        return self.demonstrations[opt_state], opt_state


    def randomizer_step(self, iteration):
        if iteration == -1:
            return 0

        else:
            if iteration < self.max_iters//4:
                return np.random.randint(len(self.env.initial_states)//4)
            elif iteration < self.max_iters//2:
                return np.random.randint(len(self.env.initial_states)//2)
            elif iteration < 3*self.max_iters//4:
                return np.random.randint(3* len(self.env.initial_states) // 4)
            return np.random.randint(len(self.env.initial_states))


    def randomizer(self, iteration):
        if iteration == -1: return 0

        cutoff = self.b*len(self.env.initial_states) + min(1, (iteration/(self.a * self.max_iters))) * (1 - self.b) * len(self.env.initial_states)
        return np.random.randint(max(1, int(cutoff)))


    def teacher_curr_state_teaching(self, iteration=-1):
        state_list = list()

        for s, trajectories in self.demonstrations.items():
            cost = 0
            for trajectory in trajectories:
                cost += self.teacher_importance_score(trajectory)

            state_list.append([cost, s])

        state_list.sort(key=lambda l:l[0], reverse=True)
        index = self.randomizer(iteration)
        opt_state = state_list[index][1]
        return self.demonstrations[opt_state], opt_state


    def learner_curr_state_teaching(self, learner, iteration=-1):
        state_list = list()

        for s, trajectories in self.demonstrations.items():
            cost = 0
            for trajectory in trajectories:
                cost += self.learner_importance_score(trajectory, learner)
            state_list.append([cost, s])

        state_list.sort(key=lambda l:l[0], reverse=True)
        index = self.randomizer(iteration)
        opt_state = state_list[index][1]
        return self.demonstrations[opt_state], opt_state


    def anticurriculum_teaching(self, learner):
        traj_list = list()

        for s, trajectories in self.demonstrations.items():
            for trajectory in trajectories:
                traj_list.append([self.importance_sampling_score(trajectory, learner), trajectory, s])

        traj_list.sort(key=lambda l:l[0])
        return [traj_list[0][1]], traj_list[0][2]


    def anticurriculum_state_teaching(self, learner):
        opt_cost = np.inf
        opt_state = -1
        
        for s, trajectories in self.demonstrations.items():
            cost = 0
            for trajectory in trajectories:
                cost += self.importance_sampling_score(trajectory, learner)
            
            if cost < opt_cost:
                opt_state = s
                opt_cost = cost

        return self.demonstrations[opt_state]
