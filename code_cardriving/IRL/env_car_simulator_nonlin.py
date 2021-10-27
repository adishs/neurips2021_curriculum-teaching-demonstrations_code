import numpy as np
from scipy import sparse
import copy


class car_simulator:

    def __init__(self, length, lanes, n_lanes, gamma):
        self.actions = {0: "straight", 1: "left", 2: "right"}
        self.n_actions = len(self.actions)

        self.road_length = length
        self.lanes = lanes
        assert lanes==5 or lanes==8, "Improper lane configuration"
        self.n_lanes = n_lanes
        self.n_states = self.road_length * 2 * self.lanes * self.n_lanes + 1
        self.gamma = gamma

        self.feature_names = ["stone", "grass", "car", "ped", "HOV", "car-in-f", "ped-in-f", "police"]
        self.n_features = len(self.feature_names)
        self.W = np.array([-1, -0.5, -5, -10, 1, -2, -5, 0])

        self.feature_vectors = self.feature_vectors()
        self.sample_features()
        self.true_reward = self.compute_reward()

        self.D_init = np.zeros((self.n_states))
        self.initial_states = []
        for i in range(self.lanes*self.n_lanes):
            self.initial_states.append(i * 2 * self.road_length)
            self.D_init[i*2*self.road_length] += (1/(self.lanes*self.n_lanes))

        self.T, self.T_dense = self._transition_matrix()
        ###
        self.compute_action_feature_matrix()


    def feature_vectors(self):
        """
        @brief: create one-hot encoded feature vectors
        """
        feature_vectors = [np.zeros(self.n_features)]
        for i in range(self.n_features-1):
            feature_vectors.append(np.zeros(self.n_features))
            feature_vectors[-1][i] = 1

        return feature_vectors


    def state_to_task(self, state):
        return (state//(self.road_length*2))%self.lanes


    def previous_state(self, state):
        prev = state - 2
        return prev


    def start_state(self, state):
        if state % (self.road_length * 2) == 0:
            return True
        return False


    def first_cells(self, state):
        if state % (self.road_length * 2) < 2:
            return True
        return False


    def terminal_state(self, state):
        if (state+2) % (self.road_length * 2) < 2 or state == self.n_states-1:
            return True
        return False


    def right_lane(self, state):
        if state%2 == 1:
            return True
        return False


    def blocked_left_lane(self, state):
        if self.right_lane(state) and (self.feature_matrix[state-1, 2]==1 or self.feature_matrix[state-1, 3]==1):
            return True
        return False


    def sample_features(self):
        """
        @brief: Sampling lanes.
        """
        self.feature_matrix = np.zeros((self.n_states, self.n_features))

        for i in range(self.feature_matrix.shape[0] - 1):
            
            if ( (i//(self.road_length*2))%self.lanes == 0):
                idx = np.random.choice(len(self.feature_vectors), 1, p=[1, 0, 0, 0, 0, 0, 0, 0])[0]
                self.feature_matrix[i] = self.feature_vectors[idx]
    
            elif ( (i//(self.road_length*2))%self.lanes == 1):
                idx = np.random.choice(len(self.feature_vectors), 1, p=[0.75, 0, 0, 0.25, 0, 0, 0, 0])[0]
                self.feature_matrix[i] = self.feature_vectors[idx]
    
            elif ( (i//(self.road_length*2))%self.lanes == 2):
                if self.right_lane(i):
                    self.feature_matrix[i] = self.feature_vectors[1]
                else:
                    idx = np.random.choice(len(self.feature_vectors), 1, p=[1, 0, 0, 0, 0, 0, 0, 0])[0]
                    self.feature_matrix[i] = self.feature_vectors[idx]
    
            elif ( (i//(self.road_length*2))%self.lanes == 3):
                idx = np.random.choice(len(self.feature_vectors), 1, p=[0.6, 0.2, 0, 0.2, 0, 0, 0, 0])[0]
                self.feature_matrix[i] = self.feature_vectors[idx]
            
            elif ( (i//(self.road_length*2))%self.lanes == 4):
                if self.right_lane(i):
                    self.feature_matrix[i] = self.feature_vectors[2]
                else:
                    idx = np.random.choice(len(self.feature_vectors), 1, p=[1, 0, 0, 0, 0, 0, 0, 0])[0]
                    self.feature_matrix[i] = self.feature_vectors[idx]
            
            elif ( (i//(self.road_length*2))%self.lanes == 5):
                idx = np.random.choice(len(self.feature_vectors), 1, p=[0.6, 0, 0.2, 0.2, 0, 0, 0, 0])[0]
                self.feature_matrix[i] = self.feature_vectors[idx]
    
            elif ( (i//(self.road_length*2))%self.lanes == 6):
                if self.right_lane(i):
                    idx = np.random.choice(len(self.feature_vectors), 1, p=[0, 0, 0.95, 0, 0.05, 0, 0, 0])[0]
                    self.feature_matrix[i] = self.feature_vectors[idx]
                else:
                    idx = np.random.choice(len(self.feature_vectors), 1, p=[0.95, 0, 0, 0, 0.05, 0, 0, 0])[0]
                    self.feature_matrix[i] = self.feature_vectors[idx]

            elif ( (i//(self.road_length*2))%self.lanes == 7):
                if self.right_lane(i):
                    self.feature_matrix[i] += self.feature_vectors[5]        
                if i%(self.road_length*2) == 4 or i%(self.road_length*2) == 12:
                    self.feature_matrix[i][-1] = 1
                    self.feature_matrix[i+1][-1] = 1

            if not self.first_cells(i):
                if self.feature_matrix[i][2] == 1:
                    self.feature_matrix[self.previous_state(i)][5] = 1
                if self.feature_matrix[i][3] == 1:
                    self.feature_matrix[self.previous_state(i)][6] = 1

        #Add pedestrian if absent in T6
        for n in range(self.n_lanes):
            flag_ped = False
            start = (6 + n*self.lanes)*2*self.road_length
            for step in range(2*self.road_length):
                if (self.feature_matrix[start+step][3] == 1):
                    flag_ped = True
                    break
            if not flag_ped:
                self.feature_matrix[start+self.road_length-1] = self.feature_vectors[4]

        return


    def _transition_matrix(self):

        transitions = np.zeros((self.n_actions, self.n_states, self.n_states))

        for a in range(self.n_actions):
            for s in range(self.n_states-1):
                front, left, right = self.next_states(s)

                if not self.terminal_state(s):
                    if a==0:
                        transitions[a, s, front] = 1
                    
                    elif self.right_lane(s):
                        if a==1:
                            transitions[a, s, left] = 1
                        else:
                            transitions[a, s, left] = 0.5
                            transitions[a, s, right] = 0.5

                    else:
                        if a==1:
                            transitions[a, s, left] = 0.5
                            transitions[a, s, right] = 0.5
                        else:
                            transitions[a, s, right] = 1

                else:
                    transitions[a, s, self.n_states-1] = 1

        T = []
        for a in range(self.n_actions):
            T.append(sparse.csr_matrix(transitions[a]))

        return T, transitions


    def next_states(self, state):
        front = state+2
        if self.right_lane(state):
            left = state+1
            right = front
        else:
            left = front
            right = front+1

        return front, left, right


    def compute_reward(self):
        reward = np.zeros((self.n_states))

        for s in range(self.n_states-1):
            if self.feature_matrix[s][-1] == 1 and self.feature_matrix[s][4] == 1:
                reward[s] = -5 + 0.5
            else:
                reward[s] = np.dot(self.W, self.feature_matrix[s]) + 0.5
        return reward


    def policy_transition_matrix(self, policy):
        T_pi = np.zeros((self.n_states, self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                T_pi[s] += policy[s, a] * self.T[a][s,:]
        return sparse.csr_matrix(np.transpose(T_pi))
        

    def reward_for_rho(self, rho):
        return np.dot(self.true_reward, rho)


    def print_lane(self, lane):
        """
        Print the given lane to understand what we have.
        """
        mask = np.array([1,2,3,4,0,5])

        masked_features = np.dot(self.feature_matrix, mask)

        print (masked_features[self.road_length*2*lane: self.road_length*2*(lane+1)].reshape(self.road_length, 2))


    ### NEW ###
    def compute_action_feature_matrix(self):
        self.action_features = np.zeros((self.n_actions, self.n_states, self.n_features))
        for a in range(self.n_actions):
            self.action_features[a] = np.matmul(self.T_dense[a,:,:], self.feature_matrix)
        return


    def compute_exp_rho_bellman(self, policy, init_dist=None, eps=1e-6):
        """
        @brief: Compute teacher's SVF using Bellman's equation.
        """
        T_pi = self.policy_transition_matrix(policy)
        rho_s = np.zeros((self.n_states))

        if init_dist is None:
            init_dist = self.D_init

        while True:
            rho_old = copy.deepcopy(rho_s)
            rho_s = init_dist + T_pi.dot(self.gamma * rho_s)
            if np.linalg.norm(rho_s - rho_old, np.inf) < eps:
                break
        
        return rho_s


    def states_for_tasks(self, tasks):
        states_list = list()

        for i in range(self.n_lanes):
            for l in range(tasks):
                states_list.append((l + i*self.lanes)*2*self.road_length)

        return states_list


    def D_init_for_tasks(self, tasks):
        D_init = np.zeros((self.n_states))
        for j in range(self.n_lanes):
            for i in range(tasks):
                D_init[(i + j*self.lanes)*2*self.road_length] += (1/(tasks*self.n_lanes))

        return D_init


    def D_init_for_lane(self, lane):
        D_init = np.zeros((self.n_states))
        for i in range(self.n_lanes):
            D_init[(lane + i*self.lanes) * 2 * self.road_length] = 1/self.n_lanes

        return D_init
