# Real World Network Model
# Script written by Cody Moser
# This script runs the Potions Task on Real World Networks
# Specify in BatchRunner whether your network is weighted or unweighted
# Data should be provided in the form of edge lists rather than adjacency matrixes
# See lines 47-55 for uploading your edge lists

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid
import networkx as nx
import pandas as pd
import numpy as np
import random
import itertools

#Function to collect the average score of model by dividing the sum of agents scores by num agents
def average_score(model):
    agent_scores = [agent.score for agent in model.schedule.agents]
    N = model.num_agents
    scores = sum(agent_scores)/N
    return scores

#Function to collect the gini of the model based on agent scores
def compute_gini(model):
    agent_wealths = [agent.score for agent in model.schedule.agents]
    x = sorted(agent_wealths)
    N = model.num_agents
    if N*sum(x) !=0:
        B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
        return (1 + (1/N) - 2*B)
    else:
        return "NA"

#Create the model
class RealNetwork(Model):
    #Itertools to count which iteration the model is in during batch runs
    id_gen = itertools.count(1)

    def __init__(self, weighted=1):
        #Set null variable to 0, crossover to 0, and the step the model is in to 0
        self.crossover = 0
        self.stage = 0
        self.uid = next(self.id_gen)

        #Create the network from the edgelists
        #Specify if the edgelist is weighted or not
        self.weighted = weighted
        if self.weighted == 1:
            self.G = nx.read_edgelist(r'..\git\Potions-Model\EdgeLists\agtaforest.txt', data=(("weight", float),))
        else:
            self.read = pd.read_csv(r'..\git\Potions-Model\EdgeLists\karate.txt')
            self.G = nx.from_pandas_edgelist(self.read, source='k1', target='k2') #Specify source and target
        self.grid = NetworkGrid(self.G)

        #Set up arrays for network summary statistics
        self.clustering = np.array([])
        self.pathlength = np.array([])
        self.num_agents = nx.number_of_nodes(self.G)

        #Set schedule - in our case, the order of agent behavior is random at each step
        self.schedule = RandomActivation(self)

        #Set up collector for export to CSV
        self.datacollector = DataCollector(
            model_reporters={"Average Score": average_score,
                             "Gini": compute_gini,
                             "NumAgents": lambda m:m.num_agents,
                             "Weighted": lambda m:m.weighted,
                             "Step": lambda m:m.stage,
                             "Path Length": lambda m: m.avgpathlength,
                             "Clustering": lambda m: m.avgclustering},
            agent_reporters={"Score": lambda _: _.score}
        )

        #Allow model to run on its own rather than step by step
        self.running = True

        #Place the appropriate agent with its properties (e.g. weights) on appropriate network position

        #Create list of nodes from the network and sort it, if weighted
        if self.weighted == 1:
            nodes_list = list(map(int,self.G.nodes()))
            nodes_list.sort()
            nodes_list = list(map(str,nodes_list))
        else:
            list_of_random_nodes = self.random.sample(list(self.G.nodes()), self.num_agents)

        #For each agent in our node list, add the appropriately numbered agent to the appropriate node
        for i in range(self.num_agents):
            #Find agent i in list of all agents (Traders)
            a = Traders(i, self)
            self.schedule.add(a)
            #Place agent i on node i
            if self.weighted == 1:
                self.grid.place_agent(a, nodes_list[i])
            else:
                self.grid.place_agent(a, list_of_random_nodes[i])
        self.running = True

    #Function to obtain topology measures from the graphs
    def topologyavg(self):
        cluster = nx.average_clustering(self.G)
        self.clustering = np.append(self.clustering, cluster)
        self.avgclustering = np.average(self.clustering)
        if nx.is_connected(self.G):
            pathl = nx.average_shortest_path_length(self.G)
            self.pathlength = np.append(self.pathlength, pathl)
            self.avgpathlength = np.average(self.pathlength)

    #Define steps in the model
    def step(self):
        self.stage = self.stage + 1
        #Collect topology statistics
        self.topologyavg()
        self.schedule.step()
        self.datacollector.collect(self)
        #End simulation when crossover is obtained
        if self.crossover == 1:
            self.running = False

class Traders(Agent):
    #Create agents
    def __init__(self, unique_id, model):
        #Create agents with a unique ID
        super().__init__(unique_id, model)
        #Give each agent the initial inventory consisting of: Potion, Trajectory, and Score
        self.inventory = np.array([
            ['a1', 'a', 6, 0],
            ['a2', 'a', 8, 0],
            ['a3', 'a', 10, 0],
            ['b1', 'b', 6, 0],
            ['b2', 'b', 8, 0],
            ['b3', 'b', 10, 0]
        ])
        #Set initial score to 0
        self.score= 0

    #Function for agents to find neighbors and obtain weights
    def get_neighborhood(self):
        #Get neighbor nodes
        self.neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        self.neighbors = self.model.grid.get_cell_list_contents(self.neighbors_nodes)
        #Get the weights between you and each neighbor stored in self.model.G and add to your yneighbor_weights
        if self.model.weighted == 1:
            self.neighbor_weights = []
            for i in self.model.G[str(self.unique_id)]:
                self.neighbor_weights.append(self.model.G[str(self.unique_id)][i]['weight'])
            #Scale weights
            self.neighbor_weights = self.neighbor_weights / np.sum(self.neighbor_weights)
            self.neighbor_weights.tolist()

    #Function to select partner in a weighted network
    def pick_partner(self):
        #Select a neighbor with probability of self.neighbor_weights
        if self.model.weighted == 1:
            self.partner = np.random.choice(self.neighbors,p=self.neighbor_weights)
        else:
            self.partner = np.random.choice(self.neighbors)
        #Index partner's node
        self.partner_nodes = self.model.grid.get_neighbors(self.partner.pos, include_center=False)
        #Obtain partner's neighbor's node
        self.partner_neighbors = self.model.grid.get_cell_list_contents(self.partner_nodes)

    def trade(self):
        #Pick a random number of ingredients, 1 or 2, to trade with neighbor
        self_ingredients = random.randint(1, 2)
        #Weigh the probability of each item in your inventory for trading with your partner
        self.weights = np.divide(self.inventory[:, 2].astype(float),self.inventory[:, 2].astype(float).sum())
        #Select items to trade with your neighbor of size self_ingredients, and with probability weights
        items_1 = (np.random.choice(self.inventory[:, 0], size=self_ingredients, replace=False, p=self.weights)).tolist()
        #Do the same for your partner
        partner_weights = np.divide(self.partner.inventory[:, 2].astype(float),self.partner.inventory[:, 2].astype(float).sum())
        items_2 = (np.random.choice(self.partner.inventory[:, 0], size=(3 - self_ingredients), replace=False,p=partner_weights)).tolist()
        #Combine your items
        self.item_set = items_1 + items_2

    #Based on the combinations, check if they add up to new innovation tiers
    def combine(self):
        if all(x in self.item_set for x in ['a1', 'a2', 'a3']):
            ingredient_1a = ['1a','a', 48, 48]
            #add ingredient 1a to inventory
            if '1a' not in self.inventory:
                self.inventory = np.vstack([self.inventory,ingredient_1a])
                #Loop through your neighbors' inventory, check if they have the item, if not, give them the ingredient
            for x in range (len(self.neighbors)):
                if '1a' not in self.neighbors[x].inventory:
                    self.neighbors[x].inventory = np.vstack([self.neighbors[x].inventory,ingredient_1a])
                #Do the same for your partner
            if '1a' not in self.partner.inventory:
                self.partner.inventory = np.vstack([self.partner.inventory,ingredient_1a])
            for x in range (len(self.partner_neighbors)):
                    if '1a' not in self.partner_neighbors[x].inventory:
                        self.partner_neighbors[x].inventory = np.vstack([self.partner_neighbors[x].inventory, ingredient_1a])

        elif all(x in self.item_set for x in ['b1', 'b2', 'b3']):
            ingredient_1b = ['1b','b', 48, 48]
            if '1b' not in self.inventory:
                self.inventory = np.vstack([self.inventory,ingredient_1b])
            for x in range (len(self.neighbors)):
                if '1b' not in self.neighbors[x].inventory:
                    self.neighbors[x].inventory = np.vstack([self.neighbors[x].inventory,ingredient_1b])
            if '1b' not in self.partner.inventory:
                self.partner.inventory = np.vstack([self.partner.inventory,ingredient_1b])
            for x in range (len(self.partner_neighbors)):
                if '1b' not in self.partner_neighbors[x].inventory:
                    self.partner_neighbors[x].inventory = np.vstack([self.partner_neighbors[x].inventory, ingredient_1b])

        elif all(x in self.item_set for x in ['1a', 'a1', 'b2']):
            ingredient_2a = ['2a','a', 109, 109]
            if '2a' not in self.inventory:
                self.inventory = np.vstack([self.inventory,ingredient_2a])
            for x in range (len(self.neighbors)):
                if '2a' not in self.neighbors[x].inventory:
                    self.neighbors[x].inventory = np.vstack([self.neighbors[x].inventory,ingredient_2a])
            if '2a' not in self.partner.inventory:
                self.partner.inventory = np.vstack([self.partner.inventory,ingredient_2a])
            for x in range (len(self.partner_neighbors)):
                if '2a' not in self.partner_neighbors[x].inventory:
                    self.partner_neighbors[x].inventory = np.vstack([self.partner_neighbors[x].inventory, ingredient_2a])

        elif all(x in self.item_set for x in['1b', 'a2', 'a3']):
            ingredient_2b = ['2b','b', 109, 109]
            if '2b' not in self.inventory:
                self.inventory = np.vstack([self.inventory,ingredient_2b])
            for x in range (len(self.neighbors)):
                if '2b' not in self.neighbors[x].inventory:
                    self.neighbors[x].inventory = np.vstack([self.neighbors[x].inventory,ingredient_2b])
            if '2b' not in self.partner.inventory:
                self.partner.inventory = np.vstack([self.partner.inventory,ingredient_2b])
            for x in range (len(self.partner_neighbors)):
                if '2b' not in self.partner_neighbors[x].inventory:
                    self.partner_neighbors[x].inventory = np.vstack([self.partner_neighbors[x].inventory, ingredient_2b])

        elif all(x in self.item_set for x in['2a', 'b2', 'a3']):
            ingredient_3a = ['3a','a', 188, 188]
            if '3a' not in self.inventory:
                self.inventory = np.vstack([self.inventory,ingredient_3a])
            for x in range (len(self.neighbors)):
                if '3a' not in self.neighbors[x].inventory:
                    self.neighbors[x].inventory = np.vstack([self.neighbors[x].inventory,ingredient_3a])
            if '3a' not in self.partner.inventory:
                self.partner.inventory = np.vstack([self.partner.inventory,ingredient_3a])
            for x in range (len(self.partner_neighbors)):
                if '3a' not in self.partner_neighbors[x].inventory:
                    self.partner_neighbors[x].inventory = np.vstack([self.partner_neighbors[x].inventory, ingredient_3a])

        elif all(x in self.item_set for x in['2b', 'b1', 'a2']):
            ingredient_3b = ['3b','b', 188, 188]
            if '3b' not in self.inventory:
                self.inventory = np.vstack([self.inventory,ingredient_3b])
            for x in range (len(self.neighbors)):
                if '3b' not in self.neighbors[x].inventory:
                    self.neighbors[x].inventory = np.vstack([self.neighbors[x].inventory,ingredient_3b])
            if '3b' not in self.partner.inventory:
                self.partner.inventory = np.vstack([self.partner.inventory,ingredient_3b])
            for x in range (len(self.partner_neighbors)):
                if '3b' not in self.partner_neighbors[x].inventory:
                    self.partner_neighbors[x].inventory = np.vstack([self.partner_neighbors[x].inventory, ingredient_3b])

        elif all(x in self.item_set for x in['3a', '3b', '2a']):
            ingredient_4a = ['4a','a', 358, 358]
            if '4a' not in self.inventory:
                self.inventory = np.vstack([self.inventory,ingredient_4a])
            for x in range (len(self.neighbors)):
                if '4a' not in self.neighbors[x].inventory:
                    self.neighbors[x].inventory = np.vstack([self.neighbors[x].inventory,ingredient_4a])
            if '4a' not in self.partner.inventory:
                self.partner.inventory = np.vstack([self.partner.inventory,ingredient_4a])
            for x in range (len(self.partner_neighbors)):
                if '4a' not in self.partner_neighbors[x].inventory:
                    self.partner_neighbors[x].inventory = np.vstack([self.partner_neighbors[x].inventory, ingredient_4a])

        elif all(x in self.item_set for x in['3b', '3a', '2b']):
            ingredient_4b = ['4b','b', 358, 358]
            if '4b' not in self.inventory:
                self.inventory = np.vstack([self.inventory,ingredient_4b])
            for x in range (len(self.neighbors)):
                if '4b' not in self.neighbors[x].inventory:
                    self.neighbors[x].inventory = np.vstack([self.neighbors[x].inventory,ingredient_4b])
            if '4b' not in self.partner.inventory:
                self.partner.inventory = np.vstack([self.partner.inventory,ingredient_4b])
            for x in range (len(self.partner_neighbors)):
                if '4b' not in self.partner_neighbors[x].inventory:
                    self.partner_neighbors[x].inventory = np.vstack([self.partner_neighbors[x].inventory, ingredient_4b])

    #Get your score
    def collectdata(self):
        #If everything is discovered, score = 716
        if all(x in self.inventory[:,0] for x in ['4a', '4b']):
            self.score = 716
        else:
        #Otherwise, score is equal to maximum item score in your inventory
            self.score = (self.inventory[:, 3].astype(float).max())
        if self.score >= 358:
            self.model.crossover = 1

    def step(self):
        self.get_neighborhood()
        if len(self.neighbors) >= 1:
            self.pick_partner()
            self.trade()
            self.combine()
            self.collectdata()