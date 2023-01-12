# Mesa Potions Task DataViz
# Script written by Cody Moser
# This script will take your random, caveman, or ring networks and visualize them in the Potions Task in an Apache server.
# Commment in or out the relevant sections under model_params to run your network
# Does not run Real World networks

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule
from mesa.visualization.modules import NetworkModule
from Model_Ring import NetworkModel #If running random import Model_Random; caveman, import Model_Caveman

#Define network portrayal from Mesa
def network_portrayal(G):
    portrayal = dict()
    portrayal["nodes"] = [
        {
            "id": node_id,
            "size": 5,
            #Display score on the actual model network
            "label": "Score:{}".format(agents[0].score),
            #Change colors based on different innovation tiers achieved
            #Tier 1: Dark Blue
            "color": "#002856" if agents[0].score < 48
            #Tier 2: Yellow
            else "#DAA900" if 48 <= agents[0].score < 109
            #Tier 3: Green
            else "#64A43A" if 109 <= agents[0].score < 188
            #Tier 4: Light Blue
            else "#99D9D9" if 188 <= agents[0].score < 358
            #Tier 5: Pink
            else "#ef7c8e" if 358<= agents[0].score < 716
            #Everything: Purple
            else "#9405db"
        }
        for (node_id, agents) in G.nodes.data("agent")
    ]
    portrayal["edges"] = [
        {"id": edge_id,"size": 3, "source": source, "target": target, "color": "#000000"}
        for edge_id, (source, target) in enumerate(G.edges)
    ]
    return portrayal

grid = NetworkModule(network_portrayal, 800, 800, library="sigma")#can also use "d3" for the library
#Chart the average score of the model
AvgScore = ChartModule(
    [{"Label": "Average Score", "Color": "Black"}], data_collector_name="datacollector"
)
#Chart the Gini of the network
MGini = ChartModule(
    [{"Label": "Gini", "Color": "Black"}], data_collector_name="datacollector"
)

model_params = {
    ####Unmute this and mute cliques and CliqueK for Ring and Random networks.
    "num_agents": UserSettableParameter(
        "slider",
        "Number of Agents",
        9,
        2,
        100,
        1,
        description="Choose how many agents to include in the model",
    ),
    ####Unmute this and num_agents and mute cliques and cliqueK for Random Networks
    # "prob_edge": UserSettableParameter(
    #     "slider",
    #     "Probability of Edge Creation",
    #     50,
    #     1,
    #     100,
    #     1,
    #     description="Select the probability that edges are created between nodes",
    # ),
    ###Unmute this and cliqueK and mute num_agents and prob_edge for Caveman Networks
    # "cliques": UserSettableParameter(
    #     "slider",
    #     "Cliques",
    #     5,
    #     1,
    #     20,
    #     1,
    #     description="Choose how many cliques are in the caveman graph",
    # ),
    # "cliqueK": UserSettableParameter(
    #     "slider",
    #     "Size of Cliques",
    #     4,
    #     3,
    #     20,
    #     1,
    #     description="Choose the size of each clique in the caveman graph",
    # ),
    "change_link": UserSettableParameter(
        "slider",
        "Probability of link alteration",
        0,
        0,
        100,
        1,
        description="The probability with which an agent creates a new link",
    ),
    "prob_diff": UserSettableParameter(
        "slider",
        "Probability of diffusion",
        100,
        0,
        100,
        1,
        description="The probability with which an individual agent receives an innovation from their neighbor",
    )
}

server = ModularServer(
    NetworkModel, [grid,AvgScore,MGini], "Network Model", model_params
)
#Launch Server
server.port = 8521
server.launch()