# Innovation-Facilitating Networks Create Inequality

This is the repository for Moser & Smaldino (2023) "Innovation-Facilitating Networks Create Inequality". The manuscript is publicly available at [].

You can find the following here:
- Python code for generating and running the Potions Task on Random, Connected Caveman, Ring, and your own real world networks! 
- Two Mesa Batchrunners: one for the Random, Caveman, and Ringer networks and one for weighted and unweighted real world networks.
- A Mesa Visualization Server: use this to visualize Random, Connected Caveman, and Ring networks playing the Potions Task 
- Edge lists for all networks in the paper
- CSV data files for data used in the paper
- An Rmarkdown file for generating plots from our paper

**For assistance, please contact the corresponding authors: Cody Moser (cmoser2@ucmerced.edu)**

## Anatomy of the repo

### Agent-based Model of the Potions Task

The model is run using the Mesa agent-based modeling library for batchruns and data visualization.

All models are listed under '/Code', beginning with Model_. Except in the case of real world networks (Model_RWNetworks) these files need not be altered to run the visualization or batchrunner.

To run the visualizer, launch Viz_Server.py and specify which network you are using in the fifth line of code and alter the model_params as specified in the comments.

To run the batchrunner, launch BatchRunner.py and specify which network you are using the second line of code. Alter the model_params as specified in the comments.

All raw data files are in `/data`, in unprocessed `csv` format, with one exception: the raw naïve listener experiment data contains identifiable information (such as email addresses), so in that case we instead included a processed, de-identified version of the dataset in `/results` (with associated processing code at `/analysis/preprocessing.R`). 

Scripts for preprocessing the data and running the analyses are in `/analysis`. This directory also contains standalone scripts for extracting acoustical data from the audio recordings; to run these, you will need a copy of the full audio corpus, which you can download from <https://doi.org/10.5281/zenodo.5525161>. Note that the audio extraction scripts will *not* be run automatically when knitting the manuscript, even if you set `full_run <- TRUE`.

Preprocessed data, interim datasets, output of models, and the like are in `/results`.

### RMarkdown Code for Generating Plots

Visualization code is in `/viz`, along with images and static data used for non-dynamic visualizations. The `/viz/figures` subdirectory contains static images produced by `figures.Rmd`, which can be regenerated with a `full_run` of `manuscript.Rmd`.

### Results

Research materials are in `/materials`, and include the protocol used at all fieldsites to collect recordings, a table of directional hypotheses for specific audio features in the preregistration, code to run the naïve listener experiment, and supplementary data and materials.

### Edge Lists for Real-World Networks Used

Code for the naïve listener experiment is available in `/materials/naive-listener`. This directory contains two separate versions of the experiment. The first, `naive-listener_Pushkin.js` is the code actually used to collect the data reported in the paper, distributed via Pushkin at <https://themusiclab.org/quizzes/ids>. This code will only run via Pushkin, so it will not execute as-is; we have posted it here for transparency.

The second version, `naive-listener_jsPsych.html`, and associated jsPsych files, is a version of the same experiment converted to run in standalone jsPsych (i.e., without Pushkin or any other software). While this code was not actually used to collect data, we made it to facilitate demonstration of the experiment (using only a fixed set of 16 recordings in the corpus, rather than drawing from the whole corpus). It can be used to understand the structure of the code that *was* used to collect data and is intended for informational/educational purposes. It is not a direct replication of the experiment reported in the paper. To try the demonstration experiment, clone this repository and open `naive-listener_jsPsych.html` in a browser.
