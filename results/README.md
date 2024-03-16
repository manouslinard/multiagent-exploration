## Proposed Methods // Folder Naming Convention

The following result folders (located in the `results` folder) contain results from experiments of different implementations of the proposed method. Here is a short description for each one:

* `New-CU-Same`: proposed approach - without hedac (also does not force different goals between agents)
* `New-CU-Hedac-Same`: proposed approach - with hedac (also does not force different goals between agents)
* `New-CU-DIFFGOAL`: proposed approach, with forcing different goals across agents
* `New-CU-HEDAC-DIFFGOAL`: proposed approach with hedac and forcing different goals across agents
* `New-CU-DIFFGOAL-PATH`: proposed approach, with forcing different goals across agents **and getting the entire path for u_jgr**. Also, calculates the utility for only the nf frontiers (not all cells).
* `New-CU-HEDAC-DIFFGOAL-PATH`: proposed approac with hedac and forcing different goals across agents **and getting the entire path for u_jgr**. Also, calculates the utility for only the nf frontiers (not all cells).
