# Ring Network Model
# Script written by Cody Moser
# This script runs the Potions Task on Ring Networks
# Specify in BatchRunner or in Viz_Server your parameters and run that file

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid
import networkx as nx
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
class NetworkModel(Model):
    #Itertools to count which iteration the model is in during batch runs
    id_gen = itertools.count(1)

    #Define initialization parameters for the network; these are also defined in the batch script
    def __init__(self, num_agents=26, change_link=0, prob_diff = 100):

        #Set crossover to 0 and the step the model is in to 0
        self.crossover = 0
        self.stage = 0
        self.uid = next(self.id_gen)

        #Set up the parameters for the network and agent behavior
        self.num_agents = num_agents
        self.change_link = 1 - (change_link/100)
        self.prob_diff = 1 - (prob_diff/100)

        #Create Ring graph from small world graph, k=2 p=0, n=num_agents
        self.G = nx.watts_strogatz_graph(n=self.num_agents, k=2, p=0)
        self.grid = NetworkGrid(self.G)

        #Set up arrays for network summary statistics
        self.clustering = np.array([])
        self.pathlength = np.array([])
        self.avgpathlength = self.pathlength
        self.initclustering = nx.average_clustering(self.G)

        #Check if graph is connected. If not, do not collect path length
        if nx.is_connected(self.G):
            self.initpathlength = nx.average_shortest_path_length(self.G)
        else:
            self.initpathlength = 'NA'

        #Set schedule - in our case, the order of agent behavior is random at each step
        self.schedule = RandomActivation(self)

        #Set up collector for export to CSV
        self.datacollector = DataCollector(
            model_reporters={"NumAgents": lambda m:m.num_agents,
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
                             "IncompleteGraph": lambda m: m.incomplete
                             #"Connectivity": lambda m:m.connectivity,
                             #"Run": track_run
                             })

        #Allow model to run on its own rather than step by step
        self.running = True

        #Place agents on each node
        nodes_list = list(self.G.nodes())
        nodes_list.sort()
        for i in range(self.num_agents):
            a = Traders(i, self)
            self.schedule.add(a)
            self.grid.place_agent(a, nodes_list[i])

    #Function to obtain topology measures from the graphs
    def topologyavg(self):
        cluster = nx.average_clustering(self.G)
        self.clustering = np.append(self.clustering, cluster)
        self.avgclustering = np.average(self.clustering)
        if nx.is_connected(self.G):
            pathl = nx.average_shortest_path_length(self.G)
            self.pathlength = np.append(self.pathlength, pathl)
            self.avgpathlength = np.average(self.pathlength)

  #Define function for dynamic link changes (agents change neighbors)
    def change_connections(self):
        #For each node in the graph:
        for i in list(self.G.nodes()):
            #Set a random value betwee 0,1
            self.chance = random.uniform(0, 1)
            #If value is larger than change_link, change neighbors
            if self.chance > self.change_link:
                #Find your non-neighbors
                self.notneighbors = list((self.G.nodes) - (self.G.neighbors(i)))
                #Remove yourself from non-neighbor list
                self.notneighbors.remove(i)
                #If non-neighbors is non-zero
                if len(self.notneighbors) > 0:
                    #If number of edges are non-zero
                    if len(self.G.edges(i)) > 0:
                        #Remove an edge
                        self.removedge = (random.choice(list(self.G.edges(i))))
                        self.G.remove_edge(*self.removedge)
                        #Add a new one
                        self.newpartner = random.choice(self.notneighbors)
                        self.G.add_edge(i,self.newpartner)

    #Define steps in the model
    def step(self):
        if self.stage == 0:
            #If graph is not connected, end simulation
            if nx.is_connected(self.G) == False:
                self.incomplete = 1
                self.running = False
            else:
                self.incomplete = 0
        self.stage = self.stage + 1
        #Collect topology statistics
        self.topologyavg()
        if self.stage > 1:
            self.change_connections()
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
        #Give each agent the initial inventory consisting of: Potion, Trajectory, Value, and Score
        self.inventory = np.array([
            ['a1','a', 6, 0],
            ['a2','a', 8, 0],
            ['a3','a', 10, 0],
            ['b1','b', 6, 0],
            ['b2','b', 8, 0],
            ['b3','b', 10, 0]
        ])
        #Set initial score to 0
        self.score= 0

    #Function for agents to find neighbors
    def get_neighborhood(self):
        neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        self.neighbors = self.model.grid.get_cell_list_contents(neighbors_nodes)

    #Function for agents to select a partner
    def pick_partner(self):
        #Select random neighbor
        self.partner = np.random.choice(self.neighbors)
        #Index partner's node
        self.partner_nodes = self.model.grid.get_neighbors(self.partner.pos, include_center=False)
        #Obtain partner's neighbor's
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
            ingredient_1a = ['1a','a', 48, 48]#15
            #add ingredient 1a to inventory
            if '1a' not in self.inventory:
                self.inventory = np.vstack([self.inventory,ingredient_1a])
                #Loop through your neighbors' inventory, check if they have the item, if not, give them the ingredient
            for x in range (len(self.neighbors)):
                if '1a' not in self.neighbors[x].inventory:
                    if random.uniform(0, 1) > self.model.prob_diff:
                        self.neighbors[x].inventory = np.vstack([self.neighbors[x].inventory,ingredient_1a])
                #Do the same for your partner
            if '1a' not in self.partner.inventory:
                self.partner.inventory = np.vstack([self.partner.inventory,ingredient_1a])
            for x in range (len(self.partner_neighbors)):
                    if '1a' not in self.partner_neighbors[x].inventory:
                        if random.uniform(0, 1) > self.model.prob_diff:
                            self.partner_neighbors[x].inventory = np.vstack([self.partner_neighbors[x].inventory, ingredient_1a])

        elif all(x in self.item_set for x in ['b1', 'b2', 'b3']):
            ingredient_1b = ['1b','b', 48, 48]
            if '1b' not in self.inventory:
                self.inventory = np.vstack([self.inventory,ingredient_1b])
            for x in range (len(self.neighbors)):
                if '1b' not in self.neighbors[x].inventory:
                    if random.uniform(0, 1) > self.model.prob_diff:
                        self.neighbors[x].inventory = np.vstack([self.neighbors[x].inventory,ingredient_1b])
            if '1b' not in self.partner.inventory:
                self.partner.inventory = np.vstack([self.partner.inventory,ingredient_1b])
            for x in range (len(self.partner_neighbors)):
                if '1b' not in self.partner_neighbors[x].inventory:
                    if random.uniform(0, 1) > self.model.prob_diff:
                        self.partner_neighbors[x].inventory = np.vstack([self.partner_neighbors[x].inventory, ingredient_1b])

        elif all(x in self.item_set for x in ['1a', 'a1', 'b2']):
            ingredient_2a = ['2a','a', 109, 109]#20
            if '2a' not in self.inventory:
                self.inventory = np.vstack([self.inventory,ingredient_2a])
            for x in range (len(self.neighbors)):
                if random.uniform(0, 1) > self.model.prob_diff:
                    if '2a' not in self.neighbors[x].inventory:
                        self.neighbors[x].inventory = np.vstack([self.neighbors[x].inventory,ingredient_2a])
            if '2a' not in self.partner.inventory:
                self.partner.inventory = np.vstack([self.partner.inventory,ingredient_2a])
            for x in range (len(self.partner_neighbors)):
                if random.uniform(0, 1) > self.model.prob_diff:
                    if '2a' not in self.partner_neighbors[x].inventory:
                        self.partner_neighbors[x].inventory = np.vstack([self.partner_neighbors[x].inventory, ingredient_2a])

        elif all(x in self.item_set for x in['1b', 'a2', 'a3']):
            ingredient_2b = ['2b','b', 109, 109]
            if '2b' not in self.inventory:
                self.inventory = np.vstack([self.inventory,ingredient_2b])
            for x in range (len(self.neighbors)):
                if '2b' not in self.neighbors[x].inventory:
                    if random.uniform(0, 1) > self.model.prob_diff:
                        self.neighbors[x].inventory = np.vstack([self.neighbors[x].inventory,ingredient_2b])
            if '2b' not in self.partner.inventory:
                self.partner.inventory = np.vstack([self.partner.inventory,ingredient_2b])
            for x in range (len(self.partner_neighbors)):
                if '2b' not in self.partner_neighbors[x].inventory:
                    if random.uniform(0, 1) > self.model.prob_diff:
                        self.partner_neighbors[x].inventory = np.vstack([self.partner_neighbors[x].inventory, ingredient_2b])

        elif all(x in self.item_set for x in['2a', 'b2', 'a3']):
            ingredient_3a = ['3a','a', 188, 188]#25
            if '3a' not in self.inventory:
                self.inventory = np.vstack([self.inventory,ingredient_3a])
            for x in range (len(self.neighbors)):
                if '3a' not in self.neighbors[x].inventory:
                    if random.uniform(0, 1) > self.model.prob_diff:
                        self.neighbors[x].inventory = np.vstack([self.neighbors[x].inventory,ingredient_3a])
            if '3a' not in self.partner.inventory:
                self.partner.inventory = np.vstack([self.partner.inventory,ingredient_3a])
            for x in range (len(self.partner_neighbors)):
                if '3a' not in self.partner_neighbors[x].inventory:
                    if random.uniform(0, 1) > self.model.prob_diff:
                        self.partner_neighbors[x].inventory = np.vstack([self.partner_neighbors[x].inventory, ingredient_3a])

        elif all(x in self.item_set for x in['2b', 'b1', 'a2']):
            ingredient_3b = ['3b','b', 188, 188]
            if '3b' not in self.inventory:
                self.inventory = np.vstack([self.inventory,ingredient_3b])
            for x in range (len(self.neighbors)):
                if '3b' not in self.neighbors[x].inventory:
                    if random.uniform(0, 1) > self.model.prob_diff:
                        self.neighbors[x].inventory = np.vstack([self.neighbors[x].inventory,ingredient_3b])
            if '3b' not in self.partner.inventory:
                self.partner.inventory = np.vstack([self.partner.inventory,ingredient_3b])
            for x in range (len(self.partner_neighbors)):
                if '3b' not in self.partner_neighbors[x].inventory:
                    if random.uniform(0, 1) > self.model.prob_diff:
                        self.partner_neighbors[x].inventory = np.vstack([self.partner_neighbors[x].inventory, ingredient_3b])

        elif all(x in self.item_set for x in['3a', '3b', '2a']):
            ingredient_4a = ['4a','a', 358, 358]#30
            if '4a' not in self.inventory:
                self.inventory = np.vstack([self.inventory,ingredient_4a])
            for x in range (len(self.neighbors)):
                if '4a' not in self.neighbors[x].inventory:
                    if random.uniform(0, 1) > self.model.prob_diff:
                        self.neighbors[x].inventory = np.vstack([self.neighbors[x].inventory,ingredient_4a])
            if '4a' not in self.partner.inventory:
                self.partner.inventory = np.vstack([self.partner.inventory,ingredient_4a])
            for x in range (len(self.partner_neighbors)):
                if '4a' not in self.partner_neighbors[x].inventory:
                    if random.uniform(0, 1) > self.model.prob_diff:
                        self.partner_neighbors[x].inventory = np.vstack([self.partner_neighbors[x].inventory, ingredient_4a])

        elif all(x in self.item_set for x in['3b', '3a', '2b']):
            ingredient_4b = ['4b','b', 358, 358]
            if '4b' not in self.inventory:
                self.inventory = np.vstack([self.inventory,ingredient_4b])
            for x in range (len(self.neighbors)):
                if '4b' not in self.neighbors[x].inventory:
                    if random.uniform(0, 1) > self.model.prob_diff:
                        self.neighbors[x].inventory = np.vstack([self.neighbors[x].inventory,ingredient_4b])
            if '4b' not in self.partner.inventory:
                self.partner.inventory = np.vstack([self.partner.inventory,ingredient_4b])
            for x in range (len(self.partner_neighbors)):
                if '4b' not in self.partner_neighbors[x].inventory:
                    if random.uniform(0, 1) > self.model.prob_diff:
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