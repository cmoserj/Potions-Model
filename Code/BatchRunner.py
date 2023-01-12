# Model Batchrunner
# Script written by Cody Moser
# This script will take your Caveman, Random, and Ring networks and run them in batches using Mesa batchrunner
# Not optimized for multiprocessing due to incompatibilities between Windows and Mesa
# See documentation for multiprocessing batchruns: https://mesa.readthedocs.io/en/main/apis/batchrunner.html

#Import Mesa batchrunner
from mesa.batchrunner import BatchRunner

from Model_Ring import NetworkModel, average_score, compute_gini #If running random import Model_Random; caveman, import Model_Caveman

#Select batch parameters
#Load network and score keeping functions
variable_params = {"num_agents": [50],
                   #"prob_edge": [50], #Edge probability for Random graph
                   #"cliqueK": [24], #Size of cliques for caveman graph
                   #"cliques":[2], #Number of cliques for caveman graph
                   "change_link": [0], #Default = 0
                   "prob_diff": [100]} #Default = 100

#Set up batch run parameters
batch_run = BatchRunner(NetworkModel,
                          variable_parameters= variable_params, #Use variables as defined above
                          iterations=300,
                          max_steps=1000,
                          model_reporters={
                              "NumAgents": lambda m:m.num_agents,
                              #"ProbEdge": lambda m:m.prob_edge, #Edge probability for Random graph
                              #"Cliques": lambda m:m.cliques, #Size of cliques for caveman graph
                              #"CliqueSize": lambda m:m.cliqueK, #Number of cliques for caveman graph
                              "ProbDiff": lambda m: m.prob_diff,
                              "ChangeLink": lambda m: m.change_link,
                              "Initial Path Length": lambda m: m.initpathlength,
                              "Path Length": lambda m: m.avgpathlength,
                              "Initial Clustering": lambda m: m.initclustering,
                              "Clustering": lambda m: m.avgclustering,
                              "Average Score": average_score,
                              "Gini": compute_gini,
                              "Step": lambda m: m.stage,
                              "Crossover": lambda m: m.crossover,
                              "IncompleteGraph": lambda m: m.incomplete},
                        display_progress=True)

#Run batches and collect data
batch_run.run_all()
modelvars_df = batch_run.get_model_vars_dataframe()
modelvars_df.to_csv(r"..\git\Potions-Model\Results\test.csv"
    r".csv")