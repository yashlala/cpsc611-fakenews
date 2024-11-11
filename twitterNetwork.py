# Load twitter follower network data
# Copyright (C) 2023  Matthew Jones

import numpy as np
import multiprocessing
from math import floor, ceil
import json


def bet_subset(subset):

    numNodes = 404719
    numEdges = 713319

    # soc-twitter-follows2.mtx is the original dataset (soc-twitter-follows.mtx from https://networkrepository.com/soc-twitter-follows.php) with the preamble removed for easier text parsing
    edgeList = np.loadtxt("soc-twitter-follows2.mtx", dtype=int)

    # create adjacency list
    adjList = [[] for _ in range(numNodes)]

    for edge in edgeList:
        adjList[edge[0] - 1].append(edge[1] - 1)
        adjList[edge[1] - 1].append(edge[0] - 1)

    centralitySubset = np.asarray([0] * numNodes)
    # Find all the geodesics starting at each node in subset
    for indx in subset:
        # print(indx)
        # List that will tell how many geodesics start at indx
        geos = np.asarray([0] * numNodes)

        # Find distances and weights
        distances = np.asarray([-1] * numNodes)
        weights = np.asarray([-1] * numNodes)
        dist = 0
        distances[indx] = 0
        weights[indx] = 1

        while dist in distances:
            templist = np.where(distances == dist)[0]
            for nindx in templist:
                for mindx in adjList[nindx]:
                    if distances[mindx] == -1:
                        distances[mindx] = dist + 1
                        weights[mindx] = weights[nindx]
                    elif distances[mindx] == dist + 1:
                        weights[mindx] += weights[nindx]
            dist += 1

        # Assign scores
        # Start at the bottom of the tree, whose distance is dist
        # Dont calculate geos through indx here because they will be
        # double counted.
        while dist > 0:
            current = np.where(distances == dist)[0]
            for nindx in current:
                geos[nindx] = 1
                temp = [
                    mindx for mindx in adjList[nindx] if distances[mindx] == dist + 1
                ]
                for mindx in temp:
                    if geos[mindx] == -1:
                        print("betweenness error")
                    geos[nindx] += geos[mindx] * weights[nindx] / weights[mindx]

            dist -= 1

        # Add the geodesic from indx to itself
        geos[indx] = 1

        centralitySubset += geos

    return centralitySubset


if __name__ == "__main__":
    popSize = 404719

    poolsize = 20

    indices = list(range(popSize))

    # subsets = [indices[0:10],indices[11:20],indices[21:30],indices[31:40],indices[41:50]]

    subsets = []
    for i in range(poolsize):
        lower = ceil(popSize * i / poolsize)
        upper = floor(popSize * (i + 1) / poolsize)
        subsets.append(indices[lower:upper])

    pool = multiprocessing.Pool(processes=poolsize)
    result1 = pool.map(bet_subset, subsets)

    centrality = np.asarray([0] * popSize)
    for result in result1:
        centrality += np.asarray(result)

    with open("twitter_betweenness_centrality.json", "w") as f:
        json.dump(centrality.tolist(), f)
