# Model Batchrunner
# Script written by Cody Moser
# This script will take your Caveman, Random, and Ring networks and run them in batches using Mesa batchrunner
# Not optimized for multiprocessing due to incompatibilities between Windows and Mesa
# See documentation for multiprocessing batchruns: https://mesa.readthedocs.io/en/main/apis/batchrunner.html

#Import Mesa batchrunner
import pandas as pd
from mesa.batchrunner import BatchRunner

from Model_Random import NetworkModel, average_score, compute_gini #If running random import Model_Random; caveman, import Model_Caveman

#Select batch parameters
#Load network and score keeping functions
variable_params = {"ranked":[0], #Determines whether Gini is calculated based on Potion scores or rank. For rank, set to 1.
                   "postcross": [0], #Determines whether simulation runs for 100 steps after crossover. For post-crossover steps, set to 1.
                   "num_agents": [50], #Num_agents for Random and Ring, mute for Caveman
                   #"prob_edge": [50], #Edge probability for Random graph, mute for Ring and Caveman
                   #"cliqueK": [24], #Size of cliques for caveman graph, mute for Ring and Random
                   #"cliques":[2], #Number of cliques for caveman graph, mute for Ring and Random
                   "change_link": [0], #Default = 0, to specify range, such as 0-100 in steps of 10; use range(0,10,110)
                   "prob_diff": [100]} #Default = 100

#Set up batch run parameters
batch_run = BatchRunner(NetworkModel,
                          variable_parameters= variable_params, #Use variables as defined above
                          iterations=1,
                          max_steps=1,
                          model_reporters={
                              "NumAgents": lambda m:m.num_agents,
                              #"ProbEdge": lambda m:m.prob_edge, #Edge probability for Random graph, mute for Ring and Caveman
                              #"Cliques": lambda m:m.cliques, #Size of cliques for caveman graph, mute for Ring and Random
                              #"CliqueSize": lambda m:m.cliqueK, #Number of cliques for caveman graph, mute for Ring and Random
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
                              "CrossStep": lambda m: m.crossed,
                              "IncompleteGraph": lambda m: m.incomplete,
                              "Ranked": lambda m: m.ranked,
                              "PostCrossover": lambda m: m.postcross},
####Agent-level metrics. Unmute for calculating centrality scores.
                        # agent_reporters={"Agent": "pos",
                        #                  "Pot3A": "Pot3A",
                        #                  "Pot3B": "Pot3B",
                        #                  "Final": "Final",
                        #                  "DegreeCent": "dcentrality",
                        #                  "BetweenCent": "bcentrality",
                        #                  "ClosenessCen": "ccentrality",
                        #                  "Neighbors": "nsize",
                        #                  "MaxScore": "score",
                        #                  "CumScore": "cum_score",
                        #                  "Potions": "pots"
                        #                  },
                        display_progress=True)

#Run batches and collect data
batch_run.run_all()
modelvars_df = batch_run.get_model_vars_dataframe()
modelvars_df.to_csv(r"..\git\Potions-Model\Results\modelsummaries.csv")

###Uncomment for stepwise model-level data:
#modelvars_df = batch_run.get_collector_model()
#modelvars_list = list(modelvars_df.values())
#for i in range(len(modelvars_list)):
#    modelvars_list[i]['Iteration'] = i + 1
#pd.concat(modelvars_list).to_csv(r"..\git\Potions-Model\Results\stepwisemodeldata.csv")

####Uncomment for stewpise agent-level data
#agentvars_df = batch_run.get_collector_agents()
#agentvars_list = list(agentvars_df.values())
#for i in range(len(agentvars_list)):
#    agentvars_list[i]['Iteration'] = i + 1
#    agentvars_list[i] = agentvars_list[i].tail((agentvars_list[i].index[-1][1])+ 1)
 #   #agentvars_list[i] = agentvars_list[i]#.tail(1) Uncomment to only receive the last step of the data!
#pd.concat(agentvars_list).to_csv(r"..\git\Potions-Model\Results\stepwiseagentdata.csv")