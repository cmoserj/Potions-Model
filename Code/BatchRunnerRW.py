# Model Batchrunner for Real World Networks
# Script written by Cody Moser
# This script will take your Real World networks and batch run them.
# Specify whether networks are weighted or unweighted
# Not optimized for multiprocessing due to incompatibilities between Windows and Mesa
# See documentation for multiprocessing batchruns: https://mesa.readthedocs.io/en/main/apis/batchrunner.html

#Import Mesa batchrunner
from mesa.batchrunner import BatchRunner

#Load network and score keeping functions
from Model_RWNetworks import RealNetwork, average_score, compute_gini

#Select batch parameters
variable_params = {"weighted": [1]} #specify whether network is weighted

#Set up batch run parameters
batch_run = BatchRunner(RealNetwork,
                          variable_parameters= variable_params,
                          iterations=100,
                          max_steps=1000,
                          model_reporters={
                              "NumAgents": lambda m:m.num_agents,
                              "Path Length": lambda m: m.avgpathlength,
                              "Clustering": lambda m: m.avgclustering,
                              "Average Score": average_score,
                              "Gini": compute_gini,
                              "Step": lambda m: m.stage,
                              "Crossover": lambda m: m.crossover},
                        display_progress=True)

#Run batches and collect data
batch_run.run_all()
modelvars_df = batch_run.get_model_vars_dataframe()
modelvars_df.to_csv(r"..\git\Potions-Model\Results\RWRun.csv"
    r".csv")