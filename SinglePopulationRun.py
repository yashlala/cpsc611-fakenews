########################################################
# Creates a random population and plays the game for a given number of
# time steps or until fixation is achieved
# Copyright (C) 2023  Matthew Jones
########################################################

# import multiprocessing
import networkx as nx
import matplotlib.pyplot as plt
from PopulationClass import Population

###################################
# Parameters to adjust
###################################
# Random Population
# n = 1000
# p = 0.01

# K Regular Population
# n = 900
# k = 5

# SW Population
# n = 60
# c = 2
# p = 1

# Scale Free Population
# n = 200
# c = 5

# Grid Population
m1 = 10
m2 = 10
n = m1 * m2
k = 8
p = 0

# All Population Structures
fcs = 1
max_simulation_time = 100

# y: if the system hasn't changed for `stabilization_time` timesteps,
#    mark the system as "stabilized" (or "in a loop", depends).
stabilization_time = 20


###################################
# Create the population
###################################


pop = Population("random", pop_size=20, p=0.4)

# Create initial strategies
pop.preset_random()
pop.add_n_factcheckers(1)

# Build your graph
graph = nx.from_pandas_edgelist(pop.edgelist, "lowindx", "highindx")

pos = nx.spring_layout(graph)  # circular layout alternate

color_map = []
for node in graph:
    if pop.players[node].real:
        color_map.append((0.0, 0.0, 0.8))
    elif pop.players[node].fake:
        color_map.append((0.8, 0.0, 0.0))
    else:
        color_map.append((0.0, 0.8, 0.0))

plt.subplots()
nx.draw(graph, pos, with_labels=True, node_color=color_map)
plt.savefig("initial-g.png", dpi=600)

###################################
# Run the simulation
###################################

# Initialization
oldlist = [True] * pop.pop_size
olderlist = [True] * pop.pop_size
newlist = pop.reals_list()

# increments once per loop.
simulation_time = 0
simulation_is_steady_state = False
count = 0
periodic_count = 0

components = []

# Run the simulation to a steady state
while not simulation_is_steady_state:
    components.append(pop.real_components())

    simulation_time += 1

    pop.update_step()

    olderlist = oldlist
    oldlist = newlist
    newlist = pop.reals_list()
    reals = pop.count_reals()

    # Detect if a strategy has completely fixated
    if reals == pop.pop_size - fcs:
        print("The real news strategy has completely fixated")
        break
    if reals == 0:
        print("The fake news strategy has completely fixated")
        break

    # Detect if the system has reached a fixed state, determine the larger strategy
    if oldlist == newlist:
        count += 1
    else:
        count = 0

    if count == stabilization_time:
        print("The system has reached a fixed state")
        if reals >= (pop.pop_size - fcs) / 2:
            print("The real news strategy has more players")
        else:
            print("The fake news strategy has more players")
        break

    # Detect if the system is in a periodic loop
    if olderlist == newlist:
        periodic_count += 1
    else:
        periodic_count = 0

    if periodic_count == stabilization_time:
        print("The system has reached a periodic loop")
        pop.update_step()
        reals += pop.count_reals()
        if reals >= pop.pop_size - fcs:
            print("The real news strategy has more players")
        else:
            print("The fake news strategy has more players")
        break

    # If we reach the time limit:
    if simulation_time == max_simulation_time:
        print("The system has not reached a fixed state")
        if reals >= (pop.pop_size - fcs) / 2:
            print("The real news strategy has more players")
        else:
            print("The fake news strategy has more players")
        break


###################################
# Graph the population again
###################################

color_map = []
for node in graph:
    if pop.players[node].real:
        color_map.append((0.0, 0.0, 0.8))
    elif pop.players[node].fake:
        color_map.append((0.8, 0.0, 0.0))
    else:
        color_map.append((0.0, 0.8, 0.0))

plt.subplots()
nx.draw(graph, pos, with_labels=True, node_color=color_map)
plt.savefig("final-g.png", dpi=600)

# ccs = [pop.clusteringcoeff(indx) for indx in range(pop.popsize)]
# elements = []
# for thing in ccs:
#     if not thing in elements:
#         elements.append(thing)
# elements.sort()
# realcount = [0]*len(elements)
# fakecount = [0]*len(elements)
# factcount = [0]*len(elements)

# for indx in range(pop.popsize):
#     cc = ccs[indx]
#     if pop.players[indx].real:
#         realcount[elements.index(cc)]+=1
#     elif pop.players[indx].fake:
#         fakecount[elements.index(cc)]+=1
#     else:
#         factcount[elements.index(cc)]+=1

# plt.subplots()
# plt.plot(realcount)
# plt.plot(fakecount)
# plt.plot(factcount)

plt.subplots()
for i in range(len(components)):
    y = components[i]
    plt.scatter([i] * len(y), y, c="C0", s=0.5)

plt.savefig("component-plot-2.png", dpi=600)

plt.subplots()
plt.plot([sum(x) for x in components])

plt.savefig("cooperator-plot-2.png", dpi=600)
