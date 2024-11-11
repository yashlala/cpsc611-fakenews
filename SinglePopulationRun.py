########################################################
# Creates a random population and plays the game for a given number of 
# time steps or until fixation is achieved
# Copyright (C) 2023  Matthew Jones
########################################################

#import multiprocessing
import networkx as nx
import matplotlib.pyplot as plt
from PopulationClass import Population

###################################
#Parameters to adjust
###################################
# Random Population
#n = 1000
#p = 0.01

# K Regular Population
#n = 900
#k = 5

# SW Population
# n = 60
# c = 2
# p = 1

# Scale Free Population
#n = 200
#c = 5

# Grid Population
m1 = 10
m2 = 10
n = m1*m2
k = 8
p = 0

# All Population Structures
fcs = 1
maxtime = 100
psdetection = 20



###################################
# Create the population
###################################


pop = Population('random', n=20, p=0.4)

# Create initial strategies
pop.preset_random()
pop.add_n_factcheckers(1)

# Build your graph
G=nx.from_pandas_edgelist(pop.edgelist, 'lowindx', 'highindx')
#pos = nx.circular_layout(G)
pos = nx.spring_layout(G)

# Get colors
color_map = []

for node in G:
    if pop.players[node].real:
        color_map.append((0.0,0.0,0.8))
    elif pop.players[node].fake: 
        color_map.append((0.8,0.0,0.0))
    else:
        color_map.append((0.0,0.8,0.0))
        
# Plot it
# plt.subplots()
# nx.draw(G, pos, with_labels=True, node_color = color_map)


###################################
# Run the simulation
###################################

# Initialization
oldlist = [True]*pop.popsize
olderlist = [True]*pop.popsize
newlist = pop.reals_list()
t = 0
steady = False
count = 0
periodic_count = 0

components = []

# Run the simulation to a steady state
while not steady:
    
    components.append(pop.real_components())
    
    t += 1
#    print(f'time = {t}')
    
    pop.update_step()
    
    olderlist = oldlist
    oldlist = newlist
    newlist = pop.reals_list()
    reals = pop.count_reals()
    
# Detect if a strategy has completely fixated
    if reals == pop.popsize - fcs:
        print('The real news strategy has completely fixated')
        steady = True
    if reals == 0:
        print('The fake news strategy has completely fixated')
        steady = True
    
# Detect if the system has reached a fixed state, determine the larger strategy
    if oldlist == newlist:
        count += 1
    else:
        count = 0
    
    if count == psdetection:
        print('The system has reached a fixed state')
        if reals >= (pop.popsize-fcs)/2:
            print('The real news strategy has more players')
        else:
            print('The fake news strategy has more players')
        steady = True
        
# Detect if the system is in a periodic loop
    if olderlist == newlist:
        periodic_count += 1
    else: 
        periodic_count = 0
        
    if periodic_count == psdetection:
        print('The system has reached a periodic loop')
        pop.update_step()
        reals += pop.count_reals()
        if reals >= pop.popsize-fcs:
            print('The real news strategy has more players')
        else:
            print('The fake news strategy has more players')
        steady = True
        
# If we reach the time limit:
    if t == maxtime:
        print('The system has not reached a fixed state')
        if reals >= (pop.popsize-fcs)/2:
            print('The real news strategy has more players')
        else:
            print('The fake news strategy has more players')
        steady = True






###################################
# Graph the population again
###################################

# Get colors
color_map = []

for node in G:
    if pop.players[node].real:
        color_map.append((0.0,0.0,0.8))
    elif pop.players[node].fake: 
        color_map.append((0.8,0.0,0.0))
    else:
        color_map.append((0.0,0.8,0.0))
        
# Plot it
# plt.subplots()
# nx.draw(G, pos, with_labels=True, node_color = color_map)


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
    plt.scatter([i]*len(y), y, c='C0', s=0.5)
    
# plt.savefig('componentPlot2.png', dpi=600)
    
plt.subplots()
plt.plot([sum(x) for x in components])

# plt.savefig('cooperatorPlot2.png', dpi=600)




