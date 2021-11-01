import sys
sys.path.append("../IRL/")
import os
import numpy as np
import argparse
from env_car_simulator_nonlin import *
from policy_gradient_teacher import *
from behavioural_cloning_learner import *
import mdp_solver_class as mdp_solver
import copy

np.set_printoptions(precision=2)


parser = argparse.ArgumentParser()
parser.add_argument('--n_states',nargs='?', const=1, type=int, default=10)
parser.add_argument('--n_lanes',nargs='?', const=1, type=int, default=5)
parser.add_argument('--gamma',nargs='?', const=1, type=float, default=0.999)
parser.add_argument('--eta',nargs='?', const=1, type=float, default=0.01)
parser.add_argument('--len_episode',nargs='?', const=1, type=int, default=10)
parser.add_argument('--num_trajectories',nargs='?', const=1, type=int, default=10)
parser.add_argument('--teacher_train_iter',nargs='?', const=1, type=int, default=1000)
parser.add_argument('--learner_train_iter',nargs='?', const=1, type=int, default=400)
parser.add_argument('--project',nargs='?', const=1, type=str, default="n")
parser.add_argument('--number',nargs='?', const=1, type=int, default=0)
parser.add_argument('--init_lanes',nargs='?', const=1, type=int, default=1)
parser.add_argument('--teacher_type',nargs='?', const=1, type=str, default="vi")
args = parser.parse_args()


def optimal_agent_reward(env):
    expert = mdp_solver.solver(env, env.true_reward, "deterministic")
    expert_rho = expert.compute_exp_rho_bellman()
    print ("Optimal expected reward = {0:0.2f}".format(env.reward_for_rho(expert_rho)))

    lane_rewards = np.zeros(env.lanes)
    for l in range(env.lanes):
        D_init = np.zeros(env.n_states)
        for i in range(env.n_lanes):
            D_init[(l + i*env.lanes)* 2 * env.road_length] = 1/env.n_lanes
        rho_lane = env.compute_exp_rho_bellman(expert.policy, D_init)
        lane_rewards[l] = env.reward_for_rho(rho_lane)

    return env.reward_for_rho(expert_rho), lane_rewards


def train_initial_tasks(env, learner, teacher, tasks, total_iter):
    print ("Reward difference on initial tasks = {}".format(np.sum(teacher.lane_rewards[:tasks] - learner.per_lane_reward()[:tasks])))

    teaching_states = env.states_for_tasks(tasks)
    for step in range(total_iter):
        for state in teaching_states:
            learner.gradient_update(teacher.demonstrations[state])

    print ("Final reward difference on initialized tasks = {}".format(np.sum(teacher.lane_rewards[:tasks] - learner.per_lane_reward()[:tasks])))
    return learner



def main():
    print ("Run {}".format(args.number))
    env = car_simulator(args.n_states, 8, args.n_lanes, args.gamma)
    
    teacher = policy_gradient_teacher(env, args.eta, args.learner_train_iter, args.len_episode, args.teacher_train_iter, args.num_trajectories, args.project=="y", args.teacher_type)
    l = behavioural_cloning_learner(env, args.eta, args.project=="y")
    l = train_initial_tasks(env, l, teacher, args.init_lanes, args.teacher_train_iter)

    learner_names = ["Agn", "Cur", "Cur-T", "Cur-L"]
    reward_curves = np.zeros((len(learner_names), 1, args.learner_train_iter+1))
    curriculum_curves = np.zeros((len(learner_names), env.lanes, args.learner_train_iter))
    learners = list()
    reward_curves[:, 0, 0] = l.compute_exp_reward()
    for i in range(len(learner_names)):
        learners.append(copy.deepcopy(l))

    print ("Starting training.")
    for iteration in range(args.learner_train_iter):
        if (iteration+1) % 100 == 0:
            print ("Learners training step [{}/{}]".format(iteration+1, args.learner_train_iter))

        #Agn teacher
        random_batch, state = teacher.random_state_teaching()
        learners[0].gradient_update(random_batch)
        curriculum_curves[0, env.state_to_task(state), iteration] = 1

        #Cur teacher
        curriculum_batch, state = teacher.curriculum_state_teaching(learners[1])
        learners[1].gradient_update(curriculum_batch)
        curriculum_curves[1, env.state_to_task(state), iteration] = 1

        #Cur-T teacher
        curriculum_batch, state = teacher.teacher_curr_state_teaching()
        learners[2].gradient_update(curriculum_batch)
        curriculum_curves[2, env.state_to_task(state), iteration] = 1

        #Cur-L teacher
        curriculum_batch, state = teacher.learner_curr_state_teaching(learners[3])
        learners[3].gradient_update(curriculum_batch)
        curriculum_curves[3, env.state_to_task(state), iteration] = 1

        #store expected rewards.
        for i in range(len(learner_names)):
            reward_curves[i, 0, iteration+1] = learners[i].compute_exp_reward()

    #Save reward arrays after training.
    path = "./results/init_lanes={}/".format(args.init_lanes) + "{}/"
    for i, name in enumerate(learner_names):
        learner_dir = path.format(name)
        reward_dir = learner_dir + "expected_reward/"
        curriculum_dir = learner_dir + "curriculum/"
        list_dirs = [reward_dir]
        list_dirs.append(curriculum_dir)
        if not os.path.isdir(reward_dir):
            for direc in list_dirs:
                os.makedirs(direc)
        
        for j, save_path in enumerate(list_dirs[:-1]):
            np.save(save_path + "array_{}".format(args.number), reward_curves[i, j])
        np.save(list_dirs[-1] + "array_{}".format(args.number), curriculum_curves[i])

    return


if __name__=="__main__":
    main()
