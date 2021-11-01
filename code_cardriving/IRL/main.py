import argparse
import os
from quadratic_learner import *
from teacher import *
from env_car_simulator_nonlin import *
import copy
import sys


def train_initial_tasks(learner, lanes, teacher, total_steps, env):
    if lanes == 0: return learner

    rho_exp = teacher.rho_for_lanes(lanes)
    print ("Initial reward diff = {}".format(env.reward_for_rho(rho_exp) - env.reward_for_rho(learner.rho_for_lanes(lanes))))
    for step in range(total_steps):
        learner.update_step(rho_exp, "lanes", lanes)
        learner.scale_eta(np.sqrt(step+2) / np.sqrt(step+1))
        if step==0 or ((step+1)%50 == 0):
            rho_learner = learner.rho_for_lanes(lanes)
            print ("Iteration [{}/{}] : Reward diff = {}, SVF diff = {}".format(step+1, total_steps, env.reward_for_rho(rho_exp) - env.reward_for_rho(rho_learner), np.linalg.norm(rho_exp - rho_learner)))

    print ("Final reward diff = {}".format(env.reward_for_rho(rho_exp) - env.reward_for_rho(learner.rho_for_lanes(lanes))))
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_lanes',nargs='?', const=1, type=int, default=5)
    parser.add_argument('--init_lanes',nargs='?', const=1, type=int, default=4)
    parser.add_argument('--gamma',nargs='?', const=1, type=float, default=0.99)
    parser.add_argument('--eta_teacher',nargs='?', const=1, type=float, default=0.5)
    parser.add_argument('--eta_learner',nargs='?', const=1, type=float, default=0.17)
    parser.add_argument('--num_trajectories',nargs='?', const=1, type=int, default=10)
    parser.add_argument('--num_episodes',nargs='?', const=1, type=int, default=10)
    parser.add_argument('--len_episodes',nargs='?', const=1, type=int, default=10)
    parser.add_argument('--batch_size',nargs='?', const=1, type=int, default=1)
    parser.add_argument('--max_iter',nargs='?', const=1, type=int, default=200)
    parser.add_argument('--avg_runs',nargs='?', const=1, type=int, default=1)
    parser.add_argument('--number',nargs='?', const=1, type=int, default=0)
    parser.add_argument('--env_name',nargs='?', const=1, type=str, default="driving")
    parser.add_argument('--update',nargs='?', const=1, type=str, default="state")
    args = parser.parse_args()

    learner_names = ["Agn", "Omn", "Scot", "Cur", "Cur-T", "Cur-L", "BBox"]

    for run in range(args.avg_runs):
        if args.env_name == "driving":
            env = car_simulator(10, 8, args.n_lanes, args.gamma)
            reward_curves = np.zeros((len(learner_names), args.max_iter+1))
            curriculum_curves = np.zeros((len(learner_names), env.lanes, args.max_iter))
        else:
            print ("Invalid Environment!!!")
            return

        teacher = Teacher(env, args.max_iter, args.num_trajectories, args.num_episodes, args.len_episodes, args.eta_teacher)
        
        l = quadratic_learner(env, args.eta_teacher, args.num_episodes, args.len_episodes)
        print ("Environment and agents created.")
        print ("Training on initial tasks.")
        train_initial_tasks(l, args.init_lanes, teacher, 500, env)
        print ("Learner knowledge initialized.")
        l.update_eta(args.eta_learner)

        learners = list()
        reward_curves[:, 0] = l.exp_reward
        for i in range(len(learner_names)):
            l_copy = copy.deepcopy(l)
            learners.append(l_copy)

        batch_states = teacher.batch_teacher()
        print ("Begin teacher-learner interaction.")
        for iteration in range(args.max_iter):
            if iteration==0 or (iteration+1)%100 == 0:
                print ("Iteration [{}/{}]".format(iteration+1, args.max_iter))

            #Agn teacher
            rho_random, states = teacher.random_teacher()
            rho_random = teacher.compute_exp_rho_state(states[0])
            learners[0].update_step(rho_random, args.update, states[0])
            curriculum_curves[0, env.state_to_task(states[0]), iteration] = 1

            #Omn teacher
            rho_imt, states = teacher.imt_teacher(learners[1])
            rho_imt = teacher.compute_exp_rho_state(states[0])
            learners[1].update_step(rho_imt, args.update, states[0])
            curriculum_curves[1, env.state_to_task(states[0]), iteration] = 1

            #SCOT teacher
            if iteration < len(batch_states):
                    state = batch_states[iteration]
            else:
                state = np.random.choice(env.initial_states)
            learners[2].update_step(teacher.compute_exp_rho_state(state), args.update, state)
            curriculum_curves[2, env.state_to_task(state), iteration] = 1

            #Cur teacher
            rho_curriculum, state = teacher.curriculum_state_teacher(learners[3])
            learners[3].update_step(rho_curriculum, args.update, state)
            curriculum_curves[3, env.state_to_task(state), iteration] = 1

            #Cur-T teacher
            rho_curriculum, state = teacher.teacher_curr_teacher()
            learners[4].update_step(rho_curriculum, args.update, state)
            curriculum_curves[4, env.state_to_task(state), iteration] = 1

            #Cur-L teacher
            rho_curriculum, state = teacher.learner_curr_teacher(learners[5])
            learners[5].update_step(rho_curriculum, args.update, state)
            curriculum_curves[5, env.state_to_task(state), iteration] = 1

            #BBox teacher
            rho_bbox, state = teacher.blackbox_state_teacher(learners[6], -1)
            learners[6].update_step(rho_bbox, args.update, state)
            curriculum_curves[6, env.state_to_task(state), iteration] = 1

            for i in range(len(learners)):
                reward_curves[i, iteration+1] = learners[i].exp_reward

        #save numpy array
        prefix = "./results/init_lanes={}/".format(args.init_lanes)
        reward_path = prefix + "reward/"
        curriculum_path = prefix + "curriculum/"
        if not os.path.isdir(reward_path):
            os.makedirs(reward_path)
            os.makedirs(curriculum_path)
        np.save(reward_path + "array_{}".format(args.number), reward_curves)
        np.save(curriculum_path + "array_{}".format(args.number), curriculum_curves)

    return


if __name__=="__main__":
    main()
