# Multi-Agent Maze Exploration

## Overview
This repository contains algorithms and experiment results for maze exploration with multiple agents, developed for my BSc thesis and 2 research papers, which are the following:

1) ["Multi-robot maze exploration using an efficient cost-utility method"](https://arxiv.org/abs/2407.14218v1)
2) ["Distributed maze exploration using multiple agents and optimal goal assignment"](https://ieeexplore.ieee.org/document/10605811)

For the 2nd paper, we have also prepared a short [video presentation](https://youtu.be/6U8a_EJ5RMM).

The algorithms are designed to efficiently navigate and explore maze environments using a swarm of agents. 

## Installation
To set up the project environment, first clone this repository and run:

```
pip install -r requirements.txt
```

## File Naming Convention
The resulting Excel files of the experiments (saved in both `results` and `results_submaps` folder) follow a specific naming convention to convey essential metadata. Both experiment folders contain subfolders with names that correspond to the algorithm that made these results.

The `results` folder corresponds to the experiment results from paper (1), whereas `results_submaps` folder contains for paper (2).

<u>**Important notes**</u>: 
- `new_cu_diffgoal_path_0.2` is the proposed method of paper (2) named CU-LVP.
- If a file name ends with an underscore followed by a float value (e.g., `new_cu_diffgoal_path_0.2`), it indicates that the examined λ value is set to the corresponding float (i.e., λ = 0.2).
- Some methods for both `results` and `results_submaps` folders contain lambda comparisons in the corresponding `lambda_comparison` subfolder. The different lambda ($\lambda$) values are denoted in the name of the subfolders/subfiles in a similar manner as denoted in the previous note.

The file with all experiments is named like so:

```
{algo}_all_{rows}x{cols}_{coverage_mode}_{agent_view}.xlsx
```

The file with the averages of the experiments is named like so:

```
{algo}_{rows}x{cols}_{coverage_mode}_{agent_view}_{num_test}.xlsx
```

Each element in the title signifies:

- `algo`: the algorithm used for the maze coverage (nearest frontier, HEDAC, etc).
- `rows`: The number of rows in the maze represented in the Excel file.
- `cols`: The number of columns in the maze represented in the Excel file.
- `coverage_mode`: Indicates whether the goal was to explore the entire maze (True or False). **Coverage mode (coverage_mode = True) was researched in my thesis and both papers (with different problem contexts each time).**
- `agent_view`: Represents the range of vision for each agent (typically set to 2 blocks).
- `num_test`: Applicable only in the files with the averages of the experiments, denoting the number of tests used for calculating the averages.

This naming convention provides a clear understanding of the contents and context of each Excel file generated by the algorithms. 
<!-- Also, if the Excel files start with `p_` that means that the experiments were run in parallel (using multiprocessing). -->

## Coverage Mode

This mode, denoted by `coverage_mode = True` in the titles of the Excel files, aims to **thoroughly explore the entire stage**. This mode was also researched in my thesis and both papers (different problem contexts each time).

The metrics used in the experiments are the ones denoted in paper *<u>Yan, Z., Fabresse, L., Laval, J., & Bouraqadi, N. (2015, September). Metrics for performance benchmarking of multi-robot exploration. In 2015 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (pp. 3407-3414). IEEE</u>*. For more information on how these are calculated, see paper or corresponding documentation in the code of this repo.

### Results Folder (`results` - Paper (1))

The `results` folder contains results from paper (1). The corresponding file columns are structured as follows:

**In the File with All Experiments:**
- `#_Agents`: Number of agents used.
- `Coverage`: Coverage percentage of the maze.
- `Total_Rounds`: Total rounds for the experiment to finish (maze exploration or reaching dead ends).
- `Expl_Cost`: Cost of maze exploration for this test.
- `Expl_Eff`: Efficiency of maze exploration for this test.
- `Avg_Round_Time`: Average round time per test (in seconds).
- `Avg_Agent_Step_Time`: Average time taken by an agent to make a move per test (in seconds).
- `Experiment_Time`: Time (seconds) until agents have explored target percentage (e.g. 100%).
- `Obs_Prob`: Obs probability of the test.
- `Test`: Test index.

**In the Averages File:**

(Includes standard deviation)

- `#_Agents`: Number of agents used.
- `Coverage`: Average coverage percentage of all experiments.
- `Avg_Total_Rounds`: Average total rounds of all experiments.
- `Avg_Expl_Cost`: Average cost of the exploration for all experiments.
- `Avg_Expl_Eff`: Average efficiency of the exploration for all experiments.
- `Std_Total_Rounds`: The standard deviation of rounds for all experiments.
- `Avg_Round_Time`: Average round time (in seconds) across all experiments.
- `Avg_Agent_Step_Time`: Average time taken by an agent to make a move (in seconds) across all experiments.
- `Experiment_Time`: Average time (seconds) across all experiments, until agents have explored target percentage (e.g. 100%).
- `Obs_Prob`: Obs probability of the test.

### Results_Submaps Folder (`results_submaps` - Paper (2))

The `results_submaps` folder was utilized for the distributed maze problem, addressed by paper (2) and shares similar columns with the files in the `results` folder. However, it extends the columns by including the `Comm_Cost` columns in both the average and all files respectively.

## Reach Goal Mode (not in thesis)

This mode was initially implemented during experimentation with various scenarios. Its objective is for the agents to reach a specified goal within the stage. Files associated with this mode are denoted by `coverage_mode = False` in their titles. The columns in these files are structured as follows:

**In the File with All Experiments:**
- `#_Agents`: Number of agents used.
- `Coverage`: Coverage percentage of the maze.
- `Total_Rounds`: Total rounds for the experiment to finish (maze exploration or reaching dead ends).
- `Avg_Rounds`: The average rounds it took for all agent to reach their goal (per test).
- `Avg_Round_Time`: Average round time per test (in seconds).
- `Finished_Agents`: The percentage of agents that reached their goal.
- `Avg_Agent_Step_Time`: Average time taken by an agent to make a move per test (in seconds).
- `Experiment_Time`: Time (seconds) until experiment completion.
- `Obs_Prob`: Obs probability of the test.
- `Test`: Test index.

**In the Averages File:**
- `#_Agents`: Number of agents used.
- `Coverage`: Average coverage percentage of all experiments.
- `Avg_Total_Rounds`: Average total rounds of all experiments.
- `Avg_Round_Time`: Average round time (in seconds) across all experiments.
- `Avg_Agent_Step_Time`: Average time taken by an agent to make a move (in seconds) across all experiments.
- `Experiment_Time`: Average time taken (in seconds) for experiment completion.
- `Obs_Prob`: Obs probability of the test.
