"""
Evaluate the optimal policy of an MDP.
"""
import numpy as np
import copy


def value_iteration(env, reward, eps=1e-6):
	"""
	@brief: Calculate optimal policy and corresponding optimal value function.
	@param env: environment class
	@param eps: threshold
	@return policy: optimal policy
	@return V: state value function
	@return Q: state-action value function
	"""
	V = np.zeros((env.n_states))
	Q = np.zeros((env.n_states, env.n_actions))
	policy = np.zeros((env.n_states, env.n_actions))

	iteration = 0
	while True:
		v_prev = copy.deepcopy(V)
		for a in range(env.n_actions):
			Q[:,a] = reward + env.gamma * env.T[a].dot(V)

		V = np.max(Q, axis=1)
		if (abs(np.linalg.norm(V - v_prev, np.inf)) < eps):
			break
		iteration += 1

	
	policy = Q - np.max(Q, axis=1)[:, None]
	policy[np.where((-1e-12 <= policy) & (policy <= 1e-12))] = 1
	policy[np.where(policy <= 0)] = 0
	policy = policy/policy.sum(axis=1)[:, None]

	return V, Q, policy


def value_iteration_soft(env, reward, eps=1e-6):
	"""
	@brief: Soft value iteration function.
	"""
	V = np.zeros((env.n_states))
	Q = np.zeros((env.n_states, env.n_actions))

	iteration = 0
	while True:
		v_prev = copy.deepcopy(V)
		for a in range(env.n_actions):
			Q[:,a] = reward + env.gamma * env.T[a].dot(V)

		V = softmax(Q, env.n_states)
		if (abs(np.linalg.norm(V - v_prev, np.inf)) < eps):
			break
		iteration += 1

	Q_copy = copy.deepcopy(Q)
	Q_copy -= Q.max(axis=1).reshape((env.n_states, 1)) #For numerical stability
	policy = np.exp(Q_copy) / np.exp(Q_copy).sum(axis=1).reshape((env.n_states, 1))
	
	return V, Q, policy


def softmax(Q, states):
	Amax = Q.max(axis=1)
	Qmod = Q - Amax.reshape((states, 1)) #For numerical stability
	return Amax + np.log(np.exp(Qmod).sum(axis=1))


def generate_episode(env, policy, len_episode, init_state=None):
	"""
	@brief: Compute discounted state visitation counts for a sampled trajectory.
	"""
	state_visitation = np.zeros((env.n_states))

	state = init_state
	if state is None:
		state = int(np.random.choice(env.initial_states, 1))

	episode = list()
	for t in range(len_episode):
		state_visitation[state] += env.gamma**t

		prob = policy[state]
		action = int(np.random.choice(np.arange(len(prob)), p=prob))
		episode.append([state, action])

		if env.T_dense.shape[0] == env.n_actions:
			next_state = env.T_dense[action, state, :]
		else:
			next_state = env.T_dense[state, :, action]
		state = int(np.random.choice(np.arange(env.n_states), p=next_state))

	return episode, state_visitation


def compute_exp_rho_sampling(env, policy, num_episode, len_episode, init_state=None):
	"""
	@brief: Compute feature expectations using monte-carlo sampling.
			Initial state can be specified otherwise is randomly picked.
	"""
	rho_s = np.zeros((env.n_states))
	for i in range(num_episode):
		_, state_visitation = generate_episode(env, policy, len_episode, init_state)
		rho_s += state_visitation

	rho_s /= num_episode
	return rho_s


def compute_exp_rho_bellman(env, policy, bellman_iter, init_dist=None, eps=1e-6):
	"""
	@brief: Compute state-action visitation freq and feature expectation 
			for given policy using Bellman's equation.
	"""
	T_pi = env.policy_transition_matrix(policy)
	if init_dist is None:
		init_dist = env.D_init
	
	rho_list = list()
	for i in range(bellman_iter):
		rho_s = np.zeros((env.n_states))
		while True:
			rho_old = copy.deepcopy(rho_s)
			rho_s = init_dist + T_pi.dot(env.gamma * rho_s)
			if np.linalg.norm(rho_s - rho_old, np.inf) < eps:
				break
		rho_list.append(rho_s)

	return np.mean(rho_list, axis=0)


def compute_value_bellman(env, policy, bellman_iter, eps=1e-6):
	T_pi = env.policy_transition_matrix(policy)

	V_list = list()
	for i in range(bellman_iter):
		V = np.zeros((env.n_states))
		# Bellman Equation
		while True:
			V_old = copy.deepcopy(V)
			V = env.true_reward + T_pi.dot(env.gamma * V_old)
			if abs(np.linalg.norm(V - V_old, np.inf)) < eps:
				break
		V_list.append(V)
	return np.mean(V_list, axis=0)
