import numpy as np
import copy


class behavioural_cloning_learner:

    def __init__(self, env, eta, project=False):
        self.env = env
        self.eta = eta
        self.radius = 10 * np.sqrt(self.env.n_features * 2)
        self.project = project

        self.theta = np.ones(self.env.n_features * 2)
        self.pi = self.compute_policy()


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


    def gradient_update(self, trajectory_set):
        for trajectory in trajectory_set:
            for s, a in trajectory:
                old_theta = copy.deepcopy(self.theta)
                self.theta += self.eta * np.append(self.env.action_features[a, s], 2*np.dot(old_theta[self.env.n_features:], self.env.action_features[a, s])*self.env.action_features[a, s])
                dist = self.action_distribution(s, self.env, old_theta)
                for action in range(self.env.n_actions):
                    self.theta -= self.eta * dist[action] * np.append(self.env.action_features[action, s], 2*np.dot(old_theta[self.env.n_features:], self.env.action_features[action, s])*self.env.action_features[action, s])

                if self.project: self.project_theta()

        self.pi = self.compute_policy()
        return


    def project_theta(self):
        if np.linalg.norm(self.theta) > self.radius:
            self.theta = self.theta * self.radius / np.linalg.norm(self.theta)
    

    def compute_exp_reward(self, D_init=None, env=None):
        if env is None:
            env = self.env

        if D_init is None:
            D_init = env.D_init

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
