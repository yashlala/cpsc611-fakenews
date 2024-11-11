########################################################
# Creates a random population and plays the game for a given number of 
# time steps or until fixation is achieved
# Copyright (C) 2023  Matthew Jones
########################################################

#import multiprocessing
import networkx as nx
import matplotlib.pyplot as plt
from PopulationClass import Population
import random as rand
import statistics as stats


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
# n = 900
# c = 8
# p = 0.03

# Scale Free Population
# n = 900
# c = 5

# Grid Population
#m1 = 30
#m2 = 30
#n = m1*m2
#k = 8
#p = 0

# All Population Structures
fcs = 200000
iterations = 2000
psdetection = 20



###################################
# Create the population
###################################

# pop = Population('smallworld', n=n, c=c, p=p)
pop = Population('twitter')


# Create initial strategies
pop.preset_random()
# Select the individuals to become fact-checkers
fclist = []

print('cent started')

#cent = list(pop.cent_eig())
#cent = list(pop.cent_between())
cent = list(pop.degree)
#cent = [rand.random() for _ in range(pop.popsize)]

print('cent done')

for i in range(pop.popsize):
    c = cent[i]
    cent[i] = [c,i]
    
modcent = sorted(cent, reverse=True)

while len(fclist)<fcs:
    val = modcent[0][0]
    tempList = []
    tempList.append(modcent[0][1])
    del modcent[0]
    
    while modcent and modcent[0][0]==val:
        tempList.append(modcent[0][1])
        del modcent[0]
        
    if len(tempList)<fcs-len(fclist):
        fclist = fclist + tempList
    else:
        addNum = fcs-len(fclist)
        fclist = fclist + rand.sample(tempList,addNum)

pop.add_list_facts(fclist)

print('init done')

# # Build your graph
# G=nx.from_pandas_edgelist(pop.edgelist, 'lowindx', 'highindx')
# pos = nx.circular_layout(G)
# #pos = nx.spring_layout(G)

# # Get colors
# color_map = []

# for node in G:
#     if pop.players[node].real:
#         color_map.append((0.0,0.0,0.8))
#     elif pop.players[node].fake: 
#         color_map.append((0.8,0.0,0.0))
#     else:
#         color_map.append((0.0,0.8,0.0))

# # Plot it
# plt.subplots()
# nx.draw(G, pos, node_color = color_map)


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

cooperators = []

# Run the simulation to a steady state
while not steady:
    x = pop.count_reals()
    print([x,t])
    cooperators.append(x)
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
    if t == iterations:
        print('The system has not reached a fixed state')
        if reals >= (pop.popsize-fcs)/2:
            print('The real news strategy has more players')
        else:
            print('The fake news strategy has more players')
        steady = True


###################################
# Graph the population again
###################################

# # Get colors
# color_map = []

# for node in G:
#     if pop.players[node].real:
#         color_map.append((0.0,0.0,0.8))
#     elif pop.players[node].fake: 
#         color_map.append((0.8,0.0,0.0))
#     else:
#         color_map.append((0.0,0.8,0.0))
        
# # Plot it
# plt.subplots()
# nx.draw(G, pos, node_color = color_map)




# # plt.subplots()
# # for i in range(len(components)):
# #     y = components[i]
# #     plt.scatter([i]*len(y), y, c='C0', s=0.5)
    
# # plt.savefig('componentPlot2.png', dpi=600)
    
# plt.subplots()
# plt.plot(cooperators)

# # plt.savefig('cooperatorPlot2.png', dpi=600)


# highDegs = [indx for indx in range(pop.popsize) if pop.degree[indx]>299]

# n1list = []
# n2list = []
# n3list = []
# n4list = []
# m1list = []
# m2list = []
# m3list = []
# m4list = []

# realCount = 0
# fakeCount = 0
# factCount = 0

# for indx in highDegs:
#     deg = len(pop.adjlist[indx])
#     if pop.players[indx].real:
#         realCount += 1
        
#         n1temp = 0
#         n2temp = 0
#         n3temp = 0
#         n4temp = 0
        
#         for nindx in pop.adjlist[indx]:
#             if pop.players[nindx].real:
#                 if len(pop.adjlist[nindx]) == 1:
#                     n1temp += 1
#                 else:
#                     n2temp += 1
#             elif pop.players[nindx].fake:
#                 n3temp += 1
#             else:
#                 n4temp += 1
        
#         n1list.append(n1temp/deg)
#         n2list.append(n2temp/deg)
#         n3list.append(n3temp/deg)
#         n4list.append(n4temp/deg)
        
#     elif pop.players[indx].fake:
#         fakeCount += 1
        
#         m1temp = 0
#         m2temp = 0
#         m3temp = 0
#         m4temp = 0
        
#         for nindx in pop.adjlist[indx]:
#             if pop.players[nindx].real:
#                 m1temp += 1
#             elif pop.players[nindx].fake:
#                 if len(pop.adjlist[nindx]) == 1:
#                     m2temp += 1
#                 else:
#                     m3temp += 1
#             else:
#                 m4temp += 1
        
#         m1list.append(m1temp/deg)
#         m2list.append(m2temp/deg)
#         m3list.append(m3temp/deg)
#         m4list.append(m4temp/deg)
        
#     elif pop.players[indx].factcheck:
#         factCount += 1
        
#     else:
#         print('error')

# n1 = stats.mean(n1list)
# n2 = stats.mean(n2list)
# n3 = stats.mean(n3list)
# n4 = stats.mean(n4list)
# m1 = stats.mean(m1list)
# m2 = stats.mean(m2list)
# m3 = stats.mean(m3list)
# m4 = stats.mean(m4list)

