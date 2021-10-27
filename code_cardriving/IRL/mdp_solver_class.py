"""
Class definition for dynamic programming MDP solver.
"""

import dp_solver_utils


class solver:

	def __init__(self, env, reward, dp_type="stochastic", bellman_iter=1, policy=None):
		self.env = env
		self.reward = reward
		self.bellman_iter = bellman_iter
		if policy is not None:
			self.policy = policy
		else:
			if dp_type == "stochastic":
				self.V, self.Q, self.policy = self.value_iteration_soft()
			else:
				self.V, self.Q, self.policy = self.value_iteration()


	def value_iteration_soft(self):
		return dp_solver_utils.value_iteration_soft(self.env, self.reward)


	def value_iteration(self):
		return dp_solver_utils.value_iteration(self.env, self.reward)


	def sample_trajectory_from_state(self, len_episode, state):
		return dp_solver_utils.generate_episode(self.env, self.policy, len_episode, state)


	def compute_exp_rho_bellman(self, init_dist=None):
		return dp_solver_utils.compute_exp_rho_bellman(self.env, self.policy, self.bellman_iter, init_dist)


	def compute_exp_rho_sampling(self, num_episode, len_episode, state=None):
		return dp_solver_utils.compute_exp_rho_sampling(self.env, self.policy, num_episode, len_episode, state)


	def compute_exp_reward(self):
		return dp_solver_utils.compute_value_bellman(self.env, self.policy, 20)
