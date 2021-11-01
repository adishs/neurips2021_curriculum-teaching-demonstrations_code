import numpy as np
import collections
import random
import mdp_solver_class as mdp_solver
from quadratic_learner import *


class Teacher:

    def __init__(self, env, max_iters, num_trajectories, num_episode, len_episode, eta, policy=None, init_states=None):
        self.env = env
        self.num_trajectories = num_trajectories
        self.num_episode = num_episode
        self.len_episode = len_episode
        self.batch_size = 1
        self.max_iters = max_iters
        self.b = 0.
        self.a = 0.8

        self.expert = mdp_solver.solver(self.env, self.env.true_reward, "deterministic", 10, policy)

        self.init_states = init_states
        if init_states is None:
            self.init_states = self.env.initial_states

        self.expert_rho = self.expert.compute_exp_rho_bellman()
        self.expert_reward = self.env.reward_for_rho(self.expert_rho)
        self.lane_reward = self.per_lane_reward()
        print ("Optimal reward of expert = {}".format(self.expert_reward))

        self.optimal_learner = quadratic_learner(self.env, eta, num_episode, len_episode)
        self.train_optimal_model()

        self.teacher_demonstrations = self.collect_trajectories()
        self.dict_rho_state = dict()
        ## Teacher-Curr ##
        self.seen_array = np.zeros(len(self.init_states))


    def train_optimal_model(self):
        total_iterations = 500

        print ("Obtaining optimal theta parameter.")
        for iteration in range(total_iterations):
            self.optimal_learner.update_step(self.expert_rho, "exp")
            self.optimal_learner.scale_eta(np.sqrt(iteration+2) / np.sqrt(iteration+1))
            if iteration==0 or ((iteration+1)%50 == 0):
                print ("Iteration [{}/{}] : Reward diff = {}, SVF diff = {}".format(iteration+1, total_iterations, self.expert_reward - self.optimal_learner.exp_reward, np.linalg.norm(self.expert_rho - self.optimal_learner.rho)))

        print ("Final reward diff = {}".format(self.expert_reward - self.optimal_learner.exp_reward))
        return


    def rho_for_lanes(self, lanes):
        D_init = self.env.D_init_for_tasks(lanes)
        rho = self.expert.compute_exp_rho_bellman(D_init)
        return rho


    def rho_task(self, task):
        if (task+1) in self.dict_rho_state:
            return self.dict_rho_state[task+1]

        D_init = np.zeros((self.env.n_states))
        for j in range(self.env.n_lanes):
            D_init[(task + j*self.env.lanes)*2*self.env.road_length] += (1/(self.env.n_lanes))

        rho = self.expert.compute_exp_rho_bellman(D_init)

        self.dict_rho_state[task+1] = rho
        return rho


    def per_lane_reward(self):
        reward_list = np.zeros((self.env.lanes))
        for l in range(self.env.lanes):
            D_init = self.env.D_init_for_lane(l)
            rho = self.expert.compute_exp_rho_bellman(D_init)
            reward_list[l] = self.env.reward_for_rho(rho)

        return reward_list


    def value_lane(self, lane):
        D_init = np.zeros((self.env.n_states))
        for k in range(self.env.n_lanes):
            D_init[(lane + k*self.env.lanes)*2*self.env.road_length] = (1/self.env.n_lanes)
        
        V_lane = np.dot(D_init, self.expert.V)
        return V_lane


    def collect_trajectories(self):
        demonstrations = collections.defaultdict(list)

        for s in self.init_states:
            t=0
            while t < self.num_trajectories:
                episode, rho = self.expert.sample_trajectory_from_state(self.len_episode, s)
                demonstrations[s].append([rho, episode])
                t += 1

        return demonstrations


    def compute_exp_rho_state(self, state):
        if state in self.dict_rho_state:
            return self.dict_rho_state[state]

        end_state = state + (2*self.env.road_length)
        mask = np.zeros((self.env.n_states))
        mask[np.arange(state, end_state)] = self.env.lanes * self.env.n_lanes
        rho_s = mask * self.expert_rho

        self.dict_rho_state[state] = rho_s
        return rho_s
    

    def random_teacher(self):
        rho = np.zeros((self.env.n_states))
        states = list()

        for i in range(self.batch_size):
            random_state = random.choice(self.init_states)
            _, rho_sample = self.expert.sample_trajectory_from_state(self.len_episode, random_state)
            rho += rho_sample
            states.append(random_state)
        return (rho / self.batch_size), states


    def imt_teacher(self, learner, iteration=-1, mode="state"):
        trajectory_cost = list()

        for state in self.init_states:
            cost = self.compute_imt_cost(learner, state, mode)
            trajectory_cost.append([cost, state])

        trajectory_cost.sort(key=lambda l: l[0])

        rho = np.zeros((self.env.n_states))
        index = self.randomizer(iteration)
        states = [trajectory_cost[index][1]]

        return (rho/self.batch_size), states


    def compute_imt_cost(self, learner, trajectory, mode):
        """
        @brief: Compute the IMT minimization objective.
        """
        lambda_diff = learner.w_t - self.optimal_learner.w_t
        if mode == "exp":
            mu_diff = np.dot(learner.rho - trajectory[0], learner.reward_gradient())
        elif mode == "lane":
            lane = self.env.state_to_task(trajectory[1][0][0])
            mu_diff = np.dot(learner.rho_task(lane) - self.rho_task(lane), learner.reward_gradient())
        else:
            mu_diff = np.dot(learner.rho_from_state(trajectory) - self.compute_exp_rho_state(trajectory), learner.reward_gradient())
        objective = (np.square(learner.eta) * np.linalg.norm(mu_diff)**2) - (2 * learner.eta * np.dot(lambda_diff, mu_diff))
        return objective
    

    def importance_sampling_score(self, trajectory, learner):
        ratio = 1.
        for s,a in trajectory[1]:
            assert learner.solver.policy[s, a] != 0, "Deterministic policy adopted. Re-train agent."
            ratio *= (self.expert.policy[s,a]/learner.solver.policy[s,a])
        return ratio


    def teacher_importance_score(self, trajectory):
        log_ratio = 0.
        for s,a in trajectory[1]:
            log_ratio += np.log(self.expert.policy[s,a])
        return log_ratio


    def learner_importance_score(self, trajectory, learner):
        log_ratio = 0.
        for s,a in trajectory[1]:
            log_ratio -= np.log(learner.solver.policy[s,a])
        return log_ratio


    def blackbox_state_teacher(self, learner, iteration=-1):
        blackbox_objective = list()

        for s in self.init_states:
            rho_s = learner.rho_from_state(s)
            diff = np.dot(rho_s - self.compute_exp_rho_state(s), self.env.true_reward)
            blackbox_objective.append([abs(diff), s])

        blackbox_objective.sort(key=lambda l:l[0], reverse=True)
        index = self.randomizer(iteration)
        return self.compute_exp_rho_state(blackbox_objective[index][1]), blackbox_objective[index][1]


    def curriculum_state_teacher(self, learner, iteration=-1):
        state_list = list()

        for s, trajectories in self.teacher_demonstrations.items():
            cost = 0
            for trajectory in trajectories:
                cost += self.importance_sampling_score(trajectory, learner)
            
            state_list.append([cost, s])
        state_list.sort(key=lambda l:l[0], reverse=True)

        index = self.randomizer(iteration)
        opt_state = state_list[index][1]
        return self.compute_exp_rho_state(opt_state), opt_state


    def teacher_curr_teacher(self, iteration=-1):
        state_list = list()
        if iteration == -1:
            if np.sum(self.seen_array) == len(self.init_states):
                self.seen_array[:] = 0

        for s, trajectories in self.teacher_demonstrations.items():
            if iteration == -1:
                index = s//(2*self.env.road_length)
                if self.seen_array[index] == 1:
                    continue
            cost = 0
            for trajectory in trajectories:
                cost += self.teacher_importance_score(trajectory)
            state_list.append([cost, s])

        state_list.sort(key=lambda l:l[0], reverse=True)
        index = self.randomizer(iteration)
        opt_state = state_list[index][1]
        if iteration == -1:
            self.seen_array[opt_state//(2*self.env.road_length)] = 1
        return self.compute_exp_rho_state(opt_state), opt_state


    def learner_curr_teacher(self, learner, iteration=-1):
        state_list = list()

        for s, trajectories in self.teacher_demonstrations.items():
            cost = 0
            for trajectory in trajectories:
                cost += self.learner_importance_score(trajectory, learner)
            
            state_list.append([cost, s])
        state_list.sort(key=lambda l:l[0], reverse=True)
        index = self.randomizer(iteration)
        opt_state = state_list[index][1]
        return self.compute_exp_rho_state(opt_state), opt_state


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


    def mu_diff_teacher(self, learner):
        opt_objective = -1
        opt_state = -1

        for s in self.init_states:
            objective = np.linalg.norm(np.dot(self.compute_exp_rho_state(s) - learner.rho_from_state(s), self.env.features) )
            if objective > opt_objective:
                opt_objective = objective
                opt_state = s

        return self.compute_exp_rho_state(opt_state), opt_state


    def anticurriculum_teacher(self, learner):
        opt_objective = np.inf
        opt_state = -1

        for s in self.init_states:
            objective = 0
            for trajectory in self.teacher_demonstrations[s]:
                objective += self.importance_sampling_function(trajectory, learner)
            if objective < opt_objective:
                opt_objective = objective
                opt_state = s

        return self.compute_exp_rho_state(opt_state), opt_state


    def batch_teacher(self):
        """
        @brief: SCOT teaching algorithm.
        """
        U = dict()
        for s in self.init_states:
            U[s] = np.dot(self.compute_exp_rho_state(s), self.env.feature_matrix)

        batch = list()
        while len(U) > 0:
            max_count = -1
            for s in self.init_states:
                if s in batch: continue
                mu_s = np.dot(self.compute_exp_rho_state(s), self.env.feature_matrix)
                states = list()
                count = 0
                for state, mu in U.items():
                    if np.all(mu == mu_s):
                        count += 1
                        states.append(state)
                if count > max_count:
                    max_count = count
                    max_state = s
                    states_to_remove = states
            batch.append(max_state)
            for s in states_to_remove:
                del U[s]

        return batch
