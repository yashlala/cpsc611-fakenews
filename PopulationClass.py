#########################################
# Creates a network with a given topology
# Copyright (C) 2023  Matthew Jones
#########################################

import pandas as pd
import random as rand
import numpy as np
import math
from scipy.special import comb
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from typing import List

from PlayerClass import Player
from PopulationParametersClass import PopulationParameters


class Population:
    def __init__(
        self,
        topology: str,
        pop_size=300,
        c=4,
        p=0.1,
        grid_width=10,
        grid_height=10,
        node_degree=8,
    ):
        """Initialize a population.

        Params:
            - `pop_size`: the number of individuals in the population.
            - `grid_width`: only used for 'grid' topology.
            - `grid_height`: only used for 'grid' topology.
            - `p`: for 'grid' toplogy: probability of adding additional edges.

            - `c`: for "scalefree" topology only. Sets "starting degree".
            - `node_degree`: only applies to 'regular' and 'grid' topology.

        TODO combine C and node_degree.
        """

        if topology == "grid":
            self._init_grid_topology(grid_width, grid_height, p, node_degree)
        elif topology == "random":
            self._init_random_topology(pop_size, p)
        elif topology == "scalefree":
            self._init_scalefree_topology(pop_size, c)
        elif topology == "regular":
            self._init_regular_topology(pop_size, num_neighbors=node_degree)
        elif topology == "smallworld":
            self._init_smallworld_topology(pop_size, neighbors=c, p=p)
        elif topology == "twitter":
            self._init_twitter_topology()
        else:
            raise ValueError("Unacceptable network topology")

    def _init_grid_topology(self, grid_width, grid_height, p, k):
        """Creates an height by width population on a rectangular grid.

        height and width must both be >= 3
        k is the degree and will be either 4 or 8 (TODO degree of what?)
        Adds additional edges with prob p for each edge in grid
        """
        if not (k == 4 or k == 8):
            raise ValueError("Degree must be 4 or 8")
        self.pop_size = grid_width * grid_height
        self.pop_params = PopulationParameters()
        self.players = []
        self.adj_list = []

        if grid_width < 3 or grid_height < 3:
            raise Exception("Grid must be at least 3x3!")

        # Create the players in the population, empty edge lists
        for indx in range(self.pop_size):
            player = Player()
            self.players.append(player)
            self.adj_list.append([])

        # Create the edgelist, adjacency list, adjacency matrix
        # self.adjlist will be used for computation and updates
        # self.edgelist will be used for visualization to pass to networkx
        # self.adjmat will be used for eigenvector centrality
        zeros = [0] * self.pop_size
        self.adjmat = np.asarray([zeros] * self.pop_size)

        templow = []
        temphigh = []

        for indx in range(self.pop_size):
            # Add the two individuals directly above and below
            self.adj_list[indx].append((indx + grid_height) % self.pop_size)
            self.adj_list[indx].append((indx - grid_height) % self.pop_size)
            # Add the individuals to the right
            if (indx + 1) % grid_height == 0:
                self.adj_list[indx].append((indx - grid_height + 1) % self.pop_size)
                if k == 8:
                    self.adj_list[indx].append(
                        (indx - 2 * grid_height + 1) % self.pop_size
                    )
                    self.adj_list[indx].append((indx + 1) % self.pop_size)
            else:
                self.adj_list[indx].append((indx + 1) % self.pop_size)
                if k == 8:
                    self.adj_list[indx].append((indx - grid_height + 1) % self.pop_size)
                    self.adj_list[indx].append((indx + grid_height + 1) % self.pop_size)
            # Add the individuals to the left
            if indx % grid_height == 0:
                self.adj_list[indx].append((indx + grid_height - 1) % self.pop_size)
                if k == 8:
                    self.adj_list[indx].append((indx - 1) % self.pop_size)
                    self.adj_list[indx].append(
                        (indx + 2 * grid_height - 1) % self.pop_size
                    )
            else:
                self.adj_list[indx].append((indx - 1) % self.pop_size)
                if k == 8:
                    self.adj_list[indx].append((indx - grid_height - 1) % self.pop_size)
                    self.adj_list[indx].append((indx + grid_height - 1) % self.pop_size)

        # Add the additional edges
        # Add edge with prob p for all self.popsize*k/2 edges in grid
        for edge in range(int(self.pop_size * k / 2)):
            r = rand.random()
            if r < p:
                new = False
                while new == False:
                    indx = rand.randint(0, self.pop_size - 1)
                    nindx = rand.randint(0, self.pop_size - 1)
                    if not indx == nindx and not indx in self.adj_list[nindx]:
                        new = True
                self.adj_list[indx].append(nindx)
                self.adj_list[nindx].append(indx)

        # Create the adjmat and edgelist
        for indx in range(self.pop_size):
            for nindx in self.adj_list[indx]:
                self.adjmat[indx, nindx] = 1
                if indx < nindx:
                    templow.append(indx)
                    temphigh.append(nindx)

        self.edgelist = pd.DataFrame()
        self.edgelist["lowindx"] = templow
        self.edgelist["highindx"] = temphigh

        # Create the degree list
        self.degree = []
        for indx in range(self.pop_size):
            self.degree.append(len(self.adj_list[indx]))

    def _init_random_topology(self, pop_size, p):
        # Creates a population on a random network.
        # The network has n individuals (vertices),
        # probability p connecting any two vertices
        self.pop_size = pop_size
        self.edge_prob = p
        self.pop_params = PopulationParameters()

        self.players = []
        self.adj_list = []

        # Create the players in the population, empty edge lists
        for indx in range(self.pop_size):
            player = Player()
            self.players.append(player)
            self.adj_list.append([])

        # Create the edgelist, adjacency list, adjacency matrix
        # self.adjlist will be used for computation and updates
        # self.edgelist will be used for visualization to pass to networkx
        # self.adjmat will be used for eigenvector centrality
        zeros = [0] * self.pop_size
        self.adjmat = np.asarray([zeros] * self.pop_size)

        templow = []
        temphigh = []
        for indx in range(self.pop_size - 1):
            for temp in range(self.pop_size - indx - 1):
                nindx = temp + indx + 1
                # Decide if there is an edge between indx and nindx
                r = rand.random()
                if r < self.edge_prob:
                    self.adj_list[indx].append(nindx)
                    self.adj_list[nindx].append(indx)
                    templow.append(indx)
                    temphigh.append(nindx)
                    self.adjmat[indx, nindx] = 1
                    self.adjmat[nindx, indx] = 1
        self.edgelist = pd.DataFrame()
        self.edgelist["lowindx"] = templow
        self.edgelist["highindx"] = temphigh

        # Create the degree list
        self.degree = []
        for indx in range(self.pop_size):
            self.degree.append(len(self.adj_list[indx]))

    def _init_scalefree_topology(self, pop_size, starting_degree):
        # Creates a population of pop_size players with a scale-free degree dist
        # each new individual has starting degree c

        self.pop_size = pop_size
        self.newdegree = starting_degree
        self.pop_params = PopulationParameters()

        self.players = []
        self.adj_list = []
        self.degree = []

        # Create the players in the population, empty edge lists
        for indx in range(self.pop_size):
            player = Player()
            self.players.append(player)
            self.adj_list.append([])
            self.degree.append(0)

        # cumulative degree list, needed to choose proportional to degree
        cdegree = [0] * self.pop_size

        # Create the edgelist, adjacency list, adjacency matrix
        # self.adjlist will be used for computation and updates
        # self.edgelist will be used for visualization to pass to networkx
        # self.adjmat will be used for eigenvector centrality
        zeros = [0] * self.pop_size
        self.adjmat = np.asarray([zeros] * self.pop_size)

        templow = []
        temphigh = []

        # Start with a small complete network
        for indx in range(self.newdegree):
            for nindx in range(indx + 1, self.newdegree):
                self.adj_list[indx].append(nindx)
                self.degree[indx] += 1
                for place in range(indx, self.pop_size):
                    cdegree[place] += 1
                self.adj_list[nindx].append(indx)
                self.degree[nindx] += 1
                for place in range(nindx, self.pop_size):
                    cdegree[place] += 1
                templow.append(indx)
                temphigh.append(nindx)

        # Now begin adding vertices
        for indx in range(self.newdegree, self.pop_size):
            edges = 0
            while edges < starting_degree:
                # find a new vertex to connect to indx
                r = rand.random()
                for nindx in range(self.pop_size):
                    if cdegree[nindx] / cdegree[self.pop_size - 1] > r:
                        # nindx is the selected vertex
                        break
                # test not already neighbors or the same vertex
                if not (indx in self.adj_list[nindx]) and (indx != nindx):
                    self.adj_list[indx].append(nindx)
                    self.degree[indx] += 1
                    for place in range(indx, self.pop_size):
                        cdegree[place] += 1
                    self.adj_list[nindx].append(indx)
                    self.degree[nindx] += 1
                    for place in range(nindx, self.pop_size):
                        cdegree[place] += 1
                    templow.append(nindx)
                    temphigh.append(indx)
                    edges += 1

        # Create the adjmat and edgelist
        for indx in range(self.pop_size):
            for nindx in self.adj_list[indx]:
                self.adjmat[indx, nindx] = 1

        self.edgelist = pd.DataFrame()
        self.edgelist["lowindx"] = templow
        self.edgelist["highindx"] = temphigh

    def _init_regular_topology(self, pop_size, num_neighbors):
        # Creates a population of n individuals
        # Each individual has k neighbors
        self.pop_size = pop_size
        self.degree = [num_neighbors] * self.pop_size
        self.pop_params = PopulationParameters()

        self.players = []
        self.adj_list = []

        # Create the players in the population
        for indx in range(self.pop_size):
            player = Player()
            self.players.append(player)

        # Create the edgelist, adjacency list, adjacency matrix
        # self.adjlist will be used for computation and updates
        # self.edgelist will be used for visualization to pass to networkx
        # self.adjmat will be used for eigenvector centrality
        zeros = [0] * self.pop_size
        self.adjmat = np.asarray([zeros] * self.pop_size)

        templow = []
        temphigh = []

        # Choose the edges, rechoose edges if not simple
        simple = False
        attempts = 0
        while simple == False:
            attempts += 1
            # print(f'Attempt number: {attempts}')
            simple = True
            self.adj_list = [[] for i in range(self.pop_size)]
            stubs = []
            for indx in range(self.pop_size):
                stubs += [indx] * num_neighbors

            while stubs:
                stub1 = stubs.pop(rand.randint(0, len(stubs) - 1))
                stub2 = stubs.pop(rand.randint(0, len(stubs) - 1))
                self.adj_list[stub1].append(stub2)
                self.adj_list[stub2].append(stub1)

            # Test if the new network is simple.
            for indx in range(self.pop_size):
                nbrs = self.adj_list[indx]
                if len(nbrs) != len(set(nbrs)):
                    simple = False
                    break
                if indx in nbrs:
                    simple = False
                    break

        # print('Simple!')

        # Create the adjacency matrix, edge list
        for indx in range(self.pop_size):
            for nindx in self.adj_list[indx]:
                self.adjmat[indx, nindx] = 1
                if indx < nindx:
                    templow.append(indx)
                    temphigh.append(nindx)

        self.edgelist = pd.DataFrame()
        self.edgelist["lowindx"] = templow
        self.edgelist["highindx"] = temphigh

    def _init_smallworld_topology(self, pop_size, neighbors, p):
        # Creates a population on a small world network
        # In the circle, each individual has c neighbors
        # For each edge in the circle (0.5nc), add random shortcut with prob p
        self.pop_size = pop_size
        self.circleneighbors = neighbors
        self.shortcutprob = p
        self.pop_params = PopulationParameters()

        self.players = []
        self.adj_list = []

        # Create the players in the population, empty edge lists
        for indx in range(self.pop_size):
            player = Player()
            self.players.append(player)
            self.adj_list.append([])

        # Create the edgelist, adjacency list, adjacency matrix
        # self.adjlist will be used for computation and updates
        # self.edgelist will be used for visualization to pass to networkx
        # self.adjmat will be used for eigenvector centrality
        zeros = [0] * self.pop_size
        self.adjmat = np.asarray([zeros] * self.pop_size)

        templow = []
        temphigh = []

        # First, the circle
        rellist = list(range(int(-neighbors / 2), 0)) + list(
            range(1, int(neighbors / 2 + 1))
        )
        for indx in range(self.pop_size):
            for rel in rellist:
                nindx = (indx + rel) % self.pop_size
                self.adj_list[indx].append(nindx)
                if indx < nindx:
                    templow.append(indx)
                    temphigh.append(nindx)

        # Now the shortcuts
        for iteration in range(int(0.5 * self.pop_size * self.circleneighbors)):
            r = rand.random()
            if r < self.shortcutprob:
                indx = rand.randint(0, self.pop_size - 1)
                nindx = rand.randint(0, self.pop_size - 1)
                if not (nindx in self.adj_list[indx]) and not (nindx == indx):
                    self.adj_list[indx].append(nindx)
                    self.adj_list[nindx].append(indx)
                    if indx < nindx:
                        templow.append(indx)
                        temphigh.append(nindx)
                    else:
                        templow.append(nindx)
                        temphigh.append(indx)

        # Create the adjmat and edgelist
        for indx in range(self.pop_size):
            for nindx in self.adj_list[indx]:
                self.adjmat[indx, nindx] = 1

        self.edgelist = pd.DataFrame()
        self.edgelist["lowindx"] = templow
        self.edgelist["highindx"] = temphigh

        # Create the degree list
        self.degree = []
        for indx in range(self.pop_size):
            self.degree.append(len(self.adj_list[indx]))

    def _init_twitter_topology(self):
        self.pop_size = 404719
        self.pop_params = PopulationParameters()
        self.edgeList = np.loadtxt("soc-twitter-follows2.mtx", dtype=int)

        self.players = []
        self.adj_list = []

        # Create the players in the population, empty edge lists
        for indx in range(self.pop_size):
            player = Player()
            self.players.append(player)
            self.adj_list.append([])

        rows = []
        cols = []

        for edge in self.edgeList:
            indx = edge[0] - 1
            nindx = edge[1] - 1
            if not nindx in self.adj_list[indx]:
                self.adj_list[indx].append(nindx)
                self.adj_list[nindx].append(indx)
                rows.append(indx)
                cols.append(nindx)
                rows.append(nindx)
                cols.append(indx)

        data = np.asarray([1] * len(rows))

        self.sparseadjmat = coo_matrix((data, (rows, cols)))
        self.sparseadjmat = self.sparseadjmat.astype(float)

        # Create the degree list
        self.degree = []
        for indx in range(self.pop_size):
            self.degree.append(len(self.adj_list[indx]))

    #########################################
    # Operations on the population
    #########################################

    # Returns the number of neighbors of [indx] playing real news
    def count_real_neighbors(self, indx):
        real_neighbors = 0

        for nindx in self.adj_list[indx]:
            if self.players[nindx].real:
                real_neighbors += 1

        return real_neighbors

    # Returns the number of neighbors of [indx] playing fake news
    def count_fake_neighbors(self, indx):
        fake_neighbors = 0

        for nindx in self.adj_list[indx]:
            if self.players[nindx].fake:
                fake_neighbors += 1

        return fake_neighbors

    # Returns the number of neighbors of [indx] playing fact-checker
    def count_factcheck_neighbors(self, indx):
        factcheck_neighbors = 0

        for nindx in self.adj_list[indx]:
            if self.players[nindx].factcheck:
                factcheck_neighbors += 1

        return factcheck_neighbors

    # Returns the number of neighbors of [indx] playing each strategy
    def count_strategy_neighbors(self, indx):
        real_neighbors = 0
        fake_neighbors = 0
        factcheck_neighbors = 0

        for nindx in self.adj_list[indx]:
            if self.players[nindx].real:
                real_neighbors += 1
            elif self.players[nindx].fake:
                fake_neighbors += 1
            elif self.players[nindx].factcheck:
                factcheck_neighbors += 1
            else:
                print("Strategy Count Error")

        return [real_neighbors, fake_neighbors, factcheck_neighbors]

    # Returns the payoff of [indx]
    def calculate_payoff(self, indx):
        payoff = 0
        [reals, fakes, factchecks] = self.count_strategy_neighbors(indx)

        # if [indx] is real
        if self.players[indx].real:
            payoff += reals * self.pop_params.payoff[0]
            payoff += fakes * self.pop_params.payoff[1]
            if self.pop_params.accuracy == 1:
                payoff += factchecks * self.pop_params.payoff[2]
            else:
                for i in range(factchecks):
                    r = rand.random()
                    if r <= self.pop_params.accuracy:
                        payoff += self.pop_params.payoff[2]
                    else:
                        payoff += self.pop_params.payoff[5]

        # if [indx] is fake
        elif self.players[indx].fake:
            payoff += reals * self.pop_params.payoff[3]
            payoff += fakes * self.pop_params.payoff[4]
            if self.pop_params.accuracy == 1:
                payoff += factchecks * self.pop_params.payoff[5]
            else:
                for i in range(factchecks):
                    r = rand.random()
                    if r <= self.pop_params.accuracy:
                        payoff += self.pop_params.payoff[5]
                    else:
                        payoff += self.pop_params.payoff[2]

        # test if [indx] is not a fact-checker
        else:
            if not self.players[indx].factcheck:
                print("Payoff Calculation Error")

        return payoff

    # Updates the strategy of each individual
    # Individuals choose to replicate a strategy proportionaly to
    # fitness from average payoff
    def update_step(self):
        # Reset all players .new
        for player in self.players:
            player.new = False

        # Create a temporary list to update from
        # True indicates update to real
        update_list = [True] * self.pop_size

        # Calculate each player's payoff
        payoffs = [0] * self.pop_size
        for indx in range(self.pop_size):
            if self.degree[indx] != 0:
                payoffs[indx] = self.calculate_payoff(indx) / self.degree[indx]

        # Update each player
        for indx in range(self.pop_size):
            if not self.players[indx].factcheck:
                # Get list of all non-fact-checker neighbors
                neighbors = []
                for nindx in self.adj_list[indx]:
                    if not self.players[nindx].factcheck:
                        neighbors.append(nindx)

                # Get the total fitness of neighbors
                totalfitness = 0
                cumulative_fit = []
                for nindx in neighbors:
                    pipay = self.pop_params.selection * payoffs[nindx]
                    totalfitness += math.exp(pipay)
                    cumulative_fit.append(totalfitness)

                # Select neighbor to copy
                choice = -1
                r = rand.random() * totalfitness
                for i in range(len(neighbors)):
                    if cumulative_fit[i] > r:
                        choice = i
                        break

                # Error Check
                if choice == -1 and neighbors:
                    print("Update Error")
                    print(indx)
                    print(neighbors)
                    print(totalfitness)
                    print(cumulative_fit)
                    print(r)

                # Find neighbor and strategy
                if neighbors:
                    chosen_neighbor = neighbors[choice]
                else:
                    chosen_neighbor = indx

                update_list[indx] = self.players[chosen_neighbor].real

        # Update player strategies with update_list
        for indx in range(self.pop_size):
            if not self.players[indx].factcheck:
                if update_list[indx]:
                    if self.players[indx].fake:
                        self.players[indx].new = True
                    self.players[indx].set_real()
                else:
                    if self.players[indx].real:
                        self.players[indx].new = True
                    self.players[indx].set_fake()

        self.avePayoffs = payoffs

    #########################################
    # Presets for the population
    #########################################

    # Set all the players to real
    def set_all_nodes_realnews(self):
        for player in self.players:
            player.set_real()
            player.new = False

    # Set all players to fake
    def set_all_nodes_fakenews(self):
        for player in self.players:
            player.set_fake()
            player.new = False

    # Set each player to real or fake with probability 1/2
    def randomize_all_nodes(self):
        for player in self.players:
            r = rand.random()
            if r <= 0.5:
                player.set_real()
            else:
                player.set_fake()
            player.new = False

    def add_factchecker_nodes(self, n: int):
        """Randomly select `n` individuals to become fact checkers"""
        # List of indices to become fact-checkers
        factcheckers = []
        while len(factcheckers) < n:
            indx = rand.randint(0, self.pop_size - 1)
            if not indx in factcheckers:
                if not self.players[indx].factcheck:
                    factcheckers.append(indx)

        for indx in factcheckers:
            self.players[indx].set_factcheck()

    def add_realnews_nodes(self, n: int):
        """Randomly select n individuals to become real news"""
        # List of indices to become real
        reals = []
        while len(reals) < n:
            indx = rand.randint(0, self.pop_size - 1)
            if not indx in reals:
                if not self.players[indx].real:
                    reals.append(indx)

        for indx in reals:
            self.players[indx].set_real()

    def add_fakenews_nodes(self, n: int):
        """Randomly select n individuals to become fake news"""
        # List of indices to become fakes
        fakes = []
        while len(fakes) < n:
            indx = rand.randint(0, self.pop_size - 1)
            if not indx in fakes:
                if not self.players[indx].fake:
                    fakes.append(indx)

        for indx in fakes:
            self.players[indx].set_fake()

    def set_factchecker_nodes(self, indxs: List[int]):
        """Takes a list of indicies and sets each of those players to fact-checker"""
        for indx in indxs:
            self.players[indx].set_factcheck()

    def set_factchecker_nodes_randomly(self, indxs: List[int], p: int):
        """Makes specific players fact-checkers with given probability"""
        for indx in indxs:
            if rand.random() < p:
                self.players[indx].set_factcheck()

    #########################################
    # Data collection for the population
    #########################################

    def get_total_realnews_count(self) -> int:
        """Count the number of members in the population who are sharing real news."""
        count = 0
        for indx in range(self.pop_size):
            if self.players[indx].real:
                count += 1
        return count

    def get_total_fakenews_count(self) -> int:
        """Count the number of members in the population who are sharing fake news."""
        count = 0
        for indx in range(self.pop_size):
            if self.players[indx].fake:
                count += 1
        return count

    def get_total_factchecker_count(self):
        """Count the number of members in the population who are fact-checking."""
        count = 0
        for indx in range(self.pop_size):
            if self.players[indx].factcheck:
                count += 1
        return count

    def count_all_strategies(self) -> List[int]:
        """Count the number of members in the population playing each strategy.

        Returns:
            [real news, fake news, fact-checking].
        """
        count_reals = 0
        count_fakes = 0
        count_factchecks = 0
        for indx in range(self.pop_size):
            if self.players[indx].real:
                count_reals += 1
            elif self.players[indx].fake:
                count_fakes += 1
            elif self.players[indx].factcheck:
                count_factchecks += 1
            else:
                print("Strategy Counter Error")
        return [count_reals, count_fakes, count_factchecks]

    def get_realnews_list(self):
        """Create a list indicating which members of the population are sharing real news.

        Returns:
            A list of boolean values where True indicates that the member at
            that index is sharing real news.
        """
        reals_list = [False] * self.pop_size
        for indx in range(self.pop_size):
            if self.players[indx].real:
                reals_list[indx] = True
        return reals_list

    # Computes the probability that two individuals have the same strategy
    # Based on the number of common neighbors they have
    # Does not include fact-checkers, but fcs do count as neighbors
    def neighbor_strat_probs(self):
        same = [0] * max(self.degree)
        dif = [0] * max(self.degree)
        totals = [0] * max(self.degree)
        probs = [0] * max(self.degree)

        for indx in range(self.pop_size):
            if not self.players[indx].factcheck:
                for nindx in range(self.pop_size):
                    if not self.players[nindx].factcheck and indx != nindx:
                        cns = self.common_neighbors(indx, nindx)
                        if self.players[indx].real == self.players[nindx].real:
                            same[cns] += 1
                        else:
                            dif[cns] += 1

        for cns in range(self.degree[0]):
            totals[cns] = same[cns] + dif[cns]
            if totals[cns] != 0:
                probs[cns] = same[cns] / totals[cns]

        return probs, totals

    def pair_probabilities(self):
        same = 0
        dif = 0

        for indx in range(self.pop_size):
            if not self.players[indx].factcheck:
                for nindx in range(self.pop_size):
                    if not self.players[nindx].factcheck and indx != nindx:
                        if self.players[indx].real == self.players[nindx].real:
                            same += 1
                        else:
                            dif += 1

        prob = same / (same + dif)

        return prob

    #########################################
    # Network Statistics
    #########################################

    # Performs a breadth-first search on the subgraph of real news players
    # Returns a list with the sizes of the components
    def real_components(self):
        sizes = []
        # The list of indices of players sharing real news
        reals = []
        for indx in range(self.pop_size):
            if self.players[indx].real:
                reals.append(indx)

        # Breadth-first search
        while reals:
            component = [reals[0]]
            reals.pop(0)
            pointer = 0
            while pointer < len(component):
                for indx in self.adj_list[component[pointer]]:
                    if indx in reals:
                        component.append(indx)
                        reals.pop(reals.index(indx))
                pointer += 1
            sizes.append(len(component))

        sizes.sort(reverse=True)
        return sizes

    # Returns the eigenvalue centrality of each node
    def cent_eig(self):

        if self.pop_size > 400000:
            eval, evecs = eigsh(self.sparseadjmat, k=1, which="LM")

            centrality = evecs

        else:
            evals, evecs = np.linalg.eigh(self.adjmat)

            # Identify if the largest eigenvalue is the first or last one
            e1 = evals[0]
            e2 = evals[self.pop_size - 1]
            if abs(e2) >= abs(e1):
                indx = self.pop_size - 1
            else:
                indx = 0

            centrality = evecs[:, indx]

        # Multiply the eigenvector by -1 if necessary
        if centrality[0] < 0:
            centrality = centrality * -1

        return centrality

    # Returns the betweenness centrality of each node
    def cent_between(self):
        centrality = np.asarray([0] * self.pop_size)
        # Find all the geodesics starting at each node
        for indx in range(self.pop_size):
            print(indx)
            # List that will tell how many geodesics start at indx
            geos = np.asarray([0] * self.pop_size)

            # Find distances and weights
            distances = np.asarray([-1] * self.pop_size)
            weights = np.asarray([-1] * self.pop_size)
            dist = 0
            distances[indx] = 0
            weights[indx] = 1

            while dist in distances:
                templist = np.where(distances == dist)[0]
                for nindx in templist:
                    for mindx in self.adj_list[nindx]:
                        if distances[mindx] == -1:
                            distances[mindx] = dist + 1
                            weights[mindx] = weights[nindx]
                        elif distances[mindx] == dist + 1:
                            weights[mindx] += weights[nindx]
                dist += 1

            self.dist = distances

            # Assign scores
            # Start at the bottom of the tree, whose distance is dist
            # Dont calculate geos through indx here because they will be
            # double counted
            print("done1")
            while dist > 0:
                print(dist)
                current = np.where(distances == dist)[0]
                farther = np.where(distances == dist + 1)[0]
                for nindx in current:
                    geos[nindx] = 1
                    temp = list(set(self.adj_list[nindx]).intersection(farther))
                    for mindx in temp:
                        if geos[mindx] == -1:
                            print("betweenness error")
                        geos[nindx] += geos[mindx] * weights[nindx] / weights[mindx]

                dist -= 1

            # Add the geodesic from indx to itself
            geos[indx] = 1

            #            print(f'index {indx}')
            #            print(geos)

            centrality += geos

        return centrality

    # Returns the local clustering coefficient
    def clusteringcoeff(self, indx):
        coefficient = 0
        indxs = self.adj_list[indx]
        for nindx in indxs:
            coefficient += len(set(indxs) & set(self.adj_list[nindx])) / 2

        if len(indxs) > 1:
            coefficient = coefficient / comb(len(indxs), 2)
        return coefficient

    # Returns the number of neighbors in common between indx and nindx
    def common_neighbors(self, indx, nindx):
        neighbors = len(set(self.adj_list[indx]) & set(self.adj_list[nindx]))

        return neighbors
