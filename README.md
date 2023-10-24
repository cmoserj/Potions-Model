# Innovation-Facilitating Networks Create Inequality

This is the repository for Moser & Smaldino (2023) "Innovation-Facilitating Networks Create Inequality". The manuscript is publicly available at https://osf.io/preprints/socarxiv/n3hc6.

You can find the following here:
- Python code for generating and running the Potions Task on Random, Connected Caveman, Ring, and your own real world networks! 
- Two Mesa Batchrunners: one for the Random, Caveman, and Ring networks and one for weighted and unweighted real world networks.
- A Mesa Visualization Server: use this to visualize Random, Connected Caveman, and Ring networks playing the Potions Task 
- Edge lists for all networks in the study
- An Rmarkdown file for generating plots from our manuscript

**For assistance, please contact the corresponding author: Cody Moser (cmoser2@ucmerced.edu)**

## Anatomy of the repo

### Agent-based Model of the Potions Task

The model is run using the Mesa agent-based modeling library for batchruns and data visualization.

All models are listed under `/Code`, beginning with `Model_`. Except in the case of real world networks (`Model_RWNetworks.py`) these files do not need to be altered to run the visualizer or batchrunner.

To run the visualizer, launch `Viz_Server.py` and specify which network you are using in the fifth line of code. Alter the `model_params` as specified in the comments.

To run the batchrunner, launch `BatchRunner.py` and specify which network you are using in the second line of code. Alter the `model_params` as specified in the comments.

For using your own real world networks, alter which network you are using in `lines 47-55` of `Model_RWNetworks.py`. For batchruns, use `BatchRunnerRW.py`.

### Edge Lists for Real World Networks Used

All edge lists are contained under `/EdgeLists`. `mercedcogsci.csv` and `karate.txt` are unweighted networks, all other networks are weighted.

### Results

Data used in the study to generate our plots and results are located in a separate Dyrad repository:
`https://doi.org/10.5061/dryad.hhmgqnknz`

### Plotting the Data

Code for generating figures in the manuscript is located in `/Code/Plots.RMD`.
