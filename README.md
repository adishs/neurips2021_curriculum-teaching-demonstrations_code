# Curriculum Design for Teaching via Demonstrations: Theory and Applications [NeurIPS'21]

This folder contains code for the paper "Curriculum Design for Teaching via Demonstrations: Theory and Applications"

The code for experiments in the car driving simulator environment are provided in the folder 'code_cardriving'. In the folder there are two additional folders, 'IRL' and 'BC' for MaxEnt-IRL and CrossEnt-BC learners respectively.

The folder 'code_shortestpath' contains code for the shortest path experiments in the grid-world environment.

The following are the steps to run the code for the different sets of experiments:

## MaxEnt-IRL

To generate results, run the following command:
`python3 main.py --init_lanes x`

Here 'x' is an integer specifying the initial knowledge of the learner. In our paper we report results for the following values of 'x':

1 - Learner has initial knowledge of T0

4 - Learner has initial knowledge of T0-T3

The '--number' flag can be used to specify the run ID. In our paper we average results over 10 runs.

After running the main file for a specific initial knowledge setting, generate the result graph with the command:
`python3 generate_plots.py --init_lanes x`

'x' is as defined above.
The path to the result graph will be './results/init_lanes=x/expected_reward_graph.pdf'.
The curriculum plots will be located in the folder './results/init_lanes=x/curriculum/'.

## CrossEnt-BC

The steps for the CrossEnt-BC learner are identical to the MaxEnt-IRL learner. The steps are repeated here for completeness.
Run the following command to generate results:
`python3 main.py --init_lanes x`

'x' is an integer specifying the initial knowledge of the learner. In our paper we report results for the following values of 'x':

1 - Learner has initial knowledge of T0

4 - Learner has initial knowledge of T0-T3

The '--number' flag is used to specify the run ID. In our paper we average results over 10 runs.

After running the main file for a specific initial knowledge setting, generate the result graphs with the command:
`python3 generate_plots.py --init_lanes x`

'x' is as defined above. 
The path to the result graph will be './results/init_lanes=x/expected_reward_graph.pdf'. 
The curriculum plots will be located in the folder './results/init_lanes=x/cur/curriculum/'.


## Shortest Path Experiments

The first step is to generate the dataset running the command:
`python3 generate_dataset.py`

Subsequently, run the following command to obtain results for individual teaching algorithms:
`python3 main.py --curr_version x`

Here 'x' is an integer specifying the teaching algorithm as follows:

0 - Agn teacher

1 - Cur teacher

2 - Cur-L teacher

3 - Cur-T teacher

The '--number' flag specifies the run ID. In our paper, results are averaged over 5 runs. Additionally, the '--gpu' flag can be used to select the gpu ID. Further flags are detailed in the code.

Obtain results for all four teaching algorithms, i.e., x = {0, 1, 2, 3}. Then run the following command to get the result graph:
`python3 plot_results.py`

The path to the result graph will be './results/test_reward_graph.pdf'.

