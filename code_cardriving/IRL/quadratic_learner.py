import numpy as np
import mdp_solver_class as mdp_solver


class quadratic_learner:

    def __init__(self, env, eta, num_episode, len_episode):
        self.env = env
        self.eta = eta
        self.w_t = np.ones((env.n_features * 2))
        self.num_episode = num_episode
        self.len_episode = len_episode
        self.radius = 10 * np.sqrt(env.n_features)

        self.quadratic_reward()
        self.solver = mdp_solver.solver(self.env, self.reward, "stochastic", 50)
        #state visitation frequency
        self.rho = self.solver.compute_exp_rho_bellman()
        self.exp_reward = self.env.reward_for_rho(self.rho)


    def quadratic_reward(self):
        self.reward = np.dot(self.env.feature_matrix, self.w_t[:self.env.n_features]) + np.dot(self.env.feature_matrix, self.w_t[self.env.n_features:])**2


    def reward_gradient(self):
        grad = np.append(self.env.feature_matrix, 2 * np.transpose(np.multiply( np.dot(self.env.feature_matrix, self.w_t[self.env.n_features:]), np.transpose(self.env.feature_matrix))), axis=1)
        return grad


    def gradient(self, rho):
        grad = np.zeros(self.env.n_features*2)
        for s in range(self.env.n_states):
            grad += rho[s] * np.append(self.env.feature_matrix[s], 2 * np.dot(self.w_t[self.env.n_features:], self.env.feature_matrix[s]) * self.env.feature_matrix[s] )
        return grad

    
    def update_step(self, rho_exp, algo="state", *args):
        if algo == "exp":
            rho_learner = self.rho
        elif algo == "lanes":
            rho_learner = self.rho_for_lanes(args[0])
        elif algo == "lane":
            rho_learner = self.rho_task(self.env.state_to_task(args[0]))
        elif algo == "state":
            rho_learner = self.rho_from_state(args[0])
        else:
            print ("Improper update algorithm!")
            raise

        self.w_t -= self.eta * (self.gradient(rho_learner) - self.gradient(rho_exp) )

        self.quadratic_reward()
        self.solver = mdp_solver.solver(self.env, self.reward, "stochastic", 50)
        #state visitation frequency
        self.rho = self.solver.compute_exp_rho_bellman()
        self.exp_reward = self.env.reward_for_rho(self.rho)


    def rho_from_state(self, state, sampling=False):
        if sampling:
            rho_s = self.solver.compute_exp_rho_sampling(self.num_episode, self.len_episode, state)
        else:
            end_state = state + (2*self.env.road_length)
            mask = np.zeros((self.env.n_states))
            mask[np.arange(state, end_state)] = self.env.lanes * self.env.n_lanes
            rho_s = mask * self.rho

        return rho_s


    def rho_task(self, task):
        D_init = np.zeros((self.env.n_states))
        for j in range(self.env.n_lanes):
            D_init[(task + j*self.env.lanes)*2*self.env.road_length] += (1/(self.env.n_lanes))

        rho = self.solver.compute_exp_rho_bellman(D_init)
        return rho


    def rho_for_lanes(self, lanes):
        D_init = self.env.D_init_for_tasks(lanes)
        rho = self.solver.compute_exp_rho_bellman(D_init)
        return rho


    def update_eta(self, eta):
        self.eta = eta


    def scale_eta(self, factor):
        self.eta /= factor


    def per_lane_reward(self):
        reward_list = np.zeros((self.env.lanes))
        for l in range(self.env.lanes):
            D_init = self.env.D_init_for_lane(l)
            rho = self.solver.compute_exp_rho_bellman(D_init)
            reward_list[l] = self.env.reward_for_rho(rho)

        return reward_list


    def value_lane(self, lane):
        D_init = np.zeros((self.env.n_states))
        for k in range(self.env.n_lanes):
            D_init[(lane + k*self.env.lanes)*2*self.env.road_length] = (1/self.env.n_lanes)
        
        V_lane = np.dot(D_init, self.solver.V)
        return V_lane
