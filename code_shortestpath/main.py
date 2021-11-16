import numpy as np
import torch
import os
from agent_class import agent
import argparse
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', nargs='?', const=1, type=str, default="./datasets/")
parser.add_argument('--result_folder', nargs='?', const=1, type=str, default="./results/")
parser.add_argument('--grid_size', nargs='?', const=1, type=int, default=6)
parser.add_argument('--in_features', nargs='?', const=1, type=int, default=7)
parser.add_argument('--lr', nargs='?', const=1, type=float, default=0.01)
parser.add_argument('--max_epoch', nargs='?', const=1, type=int, default=40)
parser.add_argument('--batch_size', nargs='?', const=1, type=int, default=32)
parser.add_argument('--task_embedding_size', nargs='?', const=1, type=int, default=512)
parser.add_argument('--gpu', nargs='?', const=1, type=int, default=0)
parser.add_argument('--curr_version', nargs='?', const=1, type=int, default=0)
parser.add_argument('--number', nargs='?', const=1, type=int, default=0)
parser.add_argument('--task_type', nargs='?', const=1, type=str, default="tsp")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dict_curr = {0:["random", "uniform"], 1:["curr", "curr"], 2:["curr", "uniform"], 3:["uniform", "curr"]}
    curr_names = ["Agn", "Cur", "Cur-L", "Cur-T"]
    if args.curr_version == 0:
        b = 1
        a = 0
    else:
        b = 0.5
        a = 0.8
    pacing_fn_dict = {"b": b, "a": a}
    #train the models.
    curr_type = dict_curr[args.curr_version]
    learner = agent(args.task_type, args.grid_size, 3, curr_type[0], curr_type[1], args.in_features, args.data_dir, args.lr, args.batch_size, args.max_epoch, **pacing_fn_dict)
    path_identifier = "{}_b={}_a={}/".format(curr_names[args.curr_version], b*10, a*10)
    learner.train_model()

    print ("Saving results.")
    results_dir = os.path.join(args.result_folder, args.task_type, path_identifier)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    np.savez(os.path.join(results_dir, "run_{}".format(args.number)),
            val_performance=learner.val_performance,
            test_performance=learner.test_performance,
            val_performance_step=learner.val_performance_step,
            test_performance_step=learner.test_performance_step)
    np.save(os.path.join(results_dir, "curriculum_{}".format(args.number)), np.array(learner.curriculum, dtype=object))
    return

if __name__=="__main__":
    main()
