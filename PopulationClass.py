#########################################
# Creates a network with a given topology
#########################################

import pandas as pd
import random as rand
import numpy as np
import math
from scipy.special import comb
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from PlayerClass import Player
from PopulationParametersClass import PopulationParameters


class Population:
    def __init__(self, topology, n=900, c=4, p=0.1, m1=30, m2=30, k=8):
        
        
        
        
        if topology=='grid':
            # Creates an m1 by m2 population on a rectangular grid
            # m1 and m2 must both be >= 3
            # k is the degree and will be either 4 or 8
            # Adds additional edges with prob p for each edge in grid
            if not (k==4 or k==8):
                raise ValueError('Degree must be 4 or 8')
            self.popsize = m1*m2
            self.parameters = PopulationParameters() 
            self.players = []
            self.adjlist = []
            
            # Create the players in the population, empty edge lists
            for indx in range(self.popsize):
                player = Player()
                self.players.append(player)
                self.adjlist.append([])
                
            # Create the edgelist, adjacency list, adjacency matrix
            #self.adjlist will be used for computation and updates
            #self.edgelist will be used for visualization to pass to networkx
            #self.adjmat will be used for eigenvector centrality
            zeros = [0]*self.popsize
            self.adjmat = np.asarray([zeros] * self.popsize)
            
            templow = []
            temphigh = []
            
            for indx in range(self.popsize):
                #Add the two individuals directly above and below
                self.adjlist[indx].append((indx+m2)%self.popsize)
                self.adjlist[indx].append((indx-m2)%self.popsize)
                #Add the individuals to the right
                if (indx+1)%m2==0:
                    self.adjlist[indx].append((indx-m2+1)%self.popsize)
                    if k==8:
                        self.adjlist[indx].append((indx-2*m2+1)%self.popsize)
                        self.adjlist[indx].append((indx+1)%self.popsize)
                else:
                    self.adjlist[indx].append((indx+1)%self.popsize)
                    if k==8:
                        self.adjlist[indx].append((indx-m2+1)%self.popsize)
                        self.adjlist[indx].append((indx+m2+1)%self.popsize)
                #Add the individuals to the left
                if indx%m2==0:
                    self.adjlist[indx].append((indx+m2-1)%self.popsize)
                    if k==8:
                        self.adjlist[indx].append((indx-1)%self.popsize)
                        self.adjlist[indx].append((indx+2*m2-1)%self.popsize)
                else:
                    self.adjlist[indx].append((indx-1)%self.popsize)
                    if k==8:
                        self.adjlist[indx].append((indx-m2-1)%self.popsize)
                        self.adjlist[indx].append((indx+m2-1)%self.popsize)
                    
            # Add the additional edges
            # Add edge with prob p for all self.popsize*k/2 edges in grid
            for edge in range(int(self.popsize*k/2)):
                r = rand.random()
                if r<p:
                    new = False
                    while new == False:
                        indx = rand.randint(0,self.popsize-1)
                        nindx = rand.randint(0,self.popsize-1)
                        if not indx==nindx and not indx in self.adjlist[nindx]:
                            new = True
                    self.adjlist[indx].append(nindx)
                    self.adjlist[nindx].append(indx)
                
            # Create the adjmat and edgelist
            for indx in range(self.popsize):
                for nindx in self.adjlist[indx]:
                    self.adjmat[indx, nindx] = 1
                    if indx < nindx:
                        templow.append(indx)
                        temphigh.append(nindx)
    
            self.edgelist = pd.DataFrame()
            self.edgelist['lowindx'] = templow
            self.edgelist['highindx'] = temphigh
    
            # Create the degree list
            self.degree = []
            for indx in range(self.popsize):
                self.degree.append(len(self.adjlist[indx]))
        
        
        
        
        elif topology=='random':
            # Creates a population on a random network.
            # The network has n individuals (vertices),
            # probability p connecting any two vertices
            self.popsize = n
            self.edgeprob = p
            self.parameters = PopulationParameters()
            
            self.players = []
            self.adjlist = []
            
            # Create the players in the population, empty edge lists
            for indx in range(self.popsize):
                player = Player()
                self.players.append(player)
                self.adjlist.append([])
                
            # Create the edgelist, adjacency list, adjacency matrix
            #self.adjlist will be used for computation and updates
            #self.edgelist will be used for visualization to pass to networkx
            #self.adjmat will be used for eigenvector centrality
            zeros = [0]*self.popsize
            self.adjmat = np.asarray([zeros] * self.popsize)
            
            templow = []
            temphigh = []
            for indx in range(self.popsize-1):
                for temp in range(self.popsize - indx - 1):
                    nindx = temp + indx + 1
                    #Decide if there is an edge between indx and nindx
                    r = rand.random()
                    if r < self.edgeprob:
                        self.adjlist[indx].append(nindx)
                        self.adjlist[nindx].append(indx)
                        templow.append(indx)
                        temphigh.append(nindx)
                        self.adjmat[indx, nindx] = 1
                        self.adjmat[nindx, indx] = 1
            self.edgelist = pd.DataFrame()
            self.edgelist['lowindx'] = templow
            self.edgelist['highindx'] = temphigh
            
            # Create the degree list
            self.degree = []
            for indx in range(self.popsize):
                self.degree.append(len(self.adjlist[indx]))
            
            
            


        elif topology=='scalefree':
            # Creates a population of n players with a scale-free degree dist
            # each new individual has starting degree c
            self.popsize = n
            self.newdegree = c
            self.parameters = PopulationParameters()
            
            self.players = []
            self.adjlist = []
            self.degree = []
            
            # Create the players in the population, empty edge lists
            for indx in range(self.popsize):
                player = Player()
                self.players.append(player)
                self.adjlist.append([])
                self.degree.append(0)
                
            # cumulative degree list, needed to choose proportional to degree
            cdegree = [0]*self.popsize
                
            # Create the edgelist, adjacency list, adjacency matrix
            #self.adjlist will be used for computation and updates
            #self.edgelist will be used for visualization to pass to networkx
            #self.adjmat will be used for eigenvector centrality
            zeros = [0]*self.popsize
            self.adjmat = np.asarray([zeros] * self.popsize)
            
            templow = []
            temphigh = []
            
            #Start with a small complete network
            for indx in range(self.newdegree):
                for nindx in range(indx+1, self.newdegree):
                    self.adjlist[indx].append(nindx)
                    self.degree[indx] += 1
                    for place in range(indx, self.popsize):
                        cdegree[place] += 1
                    self.adjlist[nindx].append(indx)
                    self.degree[nindx] += 1
                    for place in range(nindx, self.popsize):
                        cdegree[place] += 1
                    templow.append(indx)
                    temphigh.append(nindx)
            
            #Now begin adding vertices
            for indx in range(self.newdegree,self.popsize):
                edges = 0
                while edges < c:
                    #find a new vertex to connect to indx
                    r = rand.random()
                    for nindx in range(self.popsize):
                        if cdegree[nindx]/cdegree[self.popsize-1]>r:
                            # nindx is the selected vertex
                            break
                    # test not already neighbors or the same vertex
                    if not (indx in self.adjlist[nindx]) and (indx!=nindx):
                        self.adjlist[indx].append(nindx)
                        self.degree[indx] += 1
                        for place in range(indx, self.popsize):
                            cdegree[place] += 1
                        self.adjlist[nindx].append(indx)
                        self.degree[nindx] += 1
                        for place in range(nindx, self.popsize):
                            cdegree[place] += 1
                        templow.append(nindx)
                        temphigh.append(indx)
                        edges += 1
                                     
            # Create the adjmat and edgelist
            for indx in range(self.popsize):
                for nindx in self.adjlist[indx]:
                    self.adjmat[indx, nindx] = 1
    
            self.edgelist = pd.DataFrame()
            self.edgelist['lowindx'] = templow
            self.edgelist['highindx'] = temphigh
    
    
    
    

        elif topology=='regular':
            # Creates a population of n individuals
            # Each individual has k neighbors
            self.popsize = n
            self.degree = [k]*self.popsize
            self.parameters = PopulationParameters()
            
            self.players = []
            self.adjlist = []
            
            # Create the players in the population
            for indx in range(self.popsize):
                player = Player()
                self.players.append(player)
                
            # Create the edgelist, adjacency list, adjacency matrix
            #self.adjlist will be used for computation and updates
            #self.edgelist will be used for visualization to pass to networkx
            #self.adjmat will be used for eigenvector centrality
            zeros = [0]*self.popsize
            self.adjmat = np.asarray([zeros] * self.popsize)
            
            templow = []
            temphigh = []
            
            #Choose the edges, rechoose edges if not simple
            simple = False
            attempts = 0
            while simple==False:
                attempts += 1
                #print(f'Attempt number: {attempts}')
                simple = True
                self.adjlist = [ [] for i in range(self.popsize)]
                stubs = []
                for indx in range(self.popsize):
                    stubs += [indx]*k
                
                while stubs:
                    stub1 = stubs.pop(rand.randint(0,len(stubs)-1))
                    stub2 = stubs.pop(rand.randint(0,len(stubs)-1))
                    self.adjlist[stub1].append(stub2)
                    self.adjlist[stub2].append(stub1)
                
            #Test if the new network is simple.
                for indx in range(self.popsize):
                    nbrs = self.adjlist[indx]
                    if len(nbrs)!=len(set(nbrs)):
                        simple = False
                        break
                    if indx in nbrs:
                        simple = False
                        break
                    
            #print('Simple!')
                        
            #Create the adjacency matrix, edge list
            for indx in range(self.popsize):
                for nindx in self.adjlist[indx]:
                    self.adjmat[indx, nindx] = 1
                    if indx < nindx:
                        templow.append(indx)
                        temphigh.append(nindx)
            
            self.edgelist = pd.DataFrame()
            self.edgelist['lowindx'] = templow
            self.edgelist['highindx'] = temphigh
        
        
        

        elif topology=='smallworld':
            # Creates a population on a small world network
            # In the circle, each individual has c neighbors
            # For each edge in the circle (0.5nc), add random shortcut with prob p
            self.popsize = n
            self.circleneighbors = c
            self.shortcutprob = p
            self.parameters = PopulationParameters()
            
            self.players = []
            self.adjlist = []
            
            # Create the players in the population, empty edge lists
            for indx in range(self.popsize):
                player = Player()
                self.players.append(player)
                self.adjlist.append([])
                
            # Create the edgelist, adjacency list, adjacency matrix
            #self.adjlist will be used for computation and updates
            #self.edgelist will be used for visualization to pass to networkx
            #self.adjmat will be used for eigenvector centrality
            zeros = [0]*self.popsize
            self.adjmat = np.asarray([zeros] * self.popsize)
            
            templow = []
            temphigh = []
            
            # First, the circle
            rellist = list(range(int(-c/2),0)) + list(range(1,int(c/2+1)))
            for indx in range(self.popsize):
                for rel in rellist:
                    nindx = (indx + rel)%self.popsize
                    self.adjlist[indx].append(nindx)
                    if indx < nindx:
                        templow.append(indx)
                        temphigh.append(nindx)
            
            # Now the shortcuts
            for iteration in range(int(0.5*self.popsize*self.circleneighbors)):
                r = rand.random()
                if r < self.shortcutprob:
                    indx = rand.randint(0,self.popsize-1)
                    nindx = rand.randint(0,self.popsize-1)
                    if not (nindx in self.adjlist[indx]) and not (nindx == indx):
                        self.adjlist[indx].append(nindx)
                        self.adjlist[nindx].append(indx)
                        if indx < nindx:
                            templow.append(indx)
                            temphigh.append(nindx)
                        else:
                            templow.append(nindx)
                            temphigh.append(indx)
            
            # Create the adjmat and edgelist        
            for indx in range(self.popsize):
                for nindx in self.adjlist[indx]:
                    self.adjmat[indx, nindx] = 1
                    
            self.edgelist = pd.DataFrame()
            self.edgelist['lowindx'] = templow
            self.edgelist['highindx'] = temphigh
            
            # Create the degree list
            self.degree = []
            for indx in range(self.popsize):
                self.degree.append(len(self.adjlist[indx]))
                
                
                
        elif topology=='twitter':
            self.popsize = 404719
            self.parameters = PopulationParameters()
            self.edgeList = np.loadtxt('soc-twitter-follows2.mtx', dtype=int)

            self.players = []
            self.adjlist = []
            
            # Create the players in the population, empty edge lists
            for indx in range(self.popsize):
                player = Player()
                self.players.append(player)
                self.adjlist.append([])
                
            rows = []
            cols = []

            for edge in self.edgeList:
                indx = edge[0]-1
                nindx = edge[1]-1
                if not nindx in self.adjlist[indx]:
                    self.adjlist[indx].append(nindx)
                    self.adjlist[nindx].append(indx)
                    rows.append(indx)
                    cols.append(nindx)
                    rows.append(nindx)
                    cols.append(indx)
                    
            
            
            data = np.asarray([1]*len(rows))
            
            self.sparseadjmat = coo_matrix((data,(rows,cols)))
            self.sparseadjmat = self.sparseadjmat.astype(float)
                    
                
            # Create the degree list
            self.degree = []
            for indx in range(self.popsize):
                self.degree.append(len(self.adjlist[indx]))
                
        else:
            raise ValueError('Unacceptable network topology')




#########################################
#Operations on the population
#########################################

#Returns the number of neighbors of [indx] playing real news
    def count_real_neighbors(self, indx):
        real_neighbors = 0
        
        for nindx in self.adjlist[indx]:
            if self.players[nindx].real:
                real_neighbors += 1
            
        return real_neighbors

#Returns the number of neighbors of [indx] playing fake news
    def count_fake_neighbors(self, indx):
        fake_neighbors = 0
        
        for nindx in self.adjlist[indx]:
            if self.players[nindx].fake:
                fake_neighbors += 1
                
        return fake_neighbors

#Returns the number of neighbors of [indx] playing fact-checker
    def count_factcheck_neighbors(self, indx):
        factcheck_neighbors = 0
        
        for nindx in self.adjlist[indx]:
            if self.players[nindx].factcheck:
                factcheck_neighbors += 1
                
        return factcheck_neighbors
    
#Returns the number of neighbors of [indx] playing each strategy
    def count_strategy_neighbors(self, indx):
        real_neighbors = 0
        fake_neighbors = 0
        factcheck_neighbors = 0
        
        for nindx in self.adjlist[indx]:
            if self.players[nindx].real:
                real_neighbors += 1
            elif self.players[nindx].fake:
                fake_neighbors += 1
            elif self.players[nindx].factcheck:
                factcheck_neighbors += 1
            else:
                print("Strategy Count Error")
                
        return [real_neighbors, fake_neighbors, factcheck_neighbors]
    
#Returns the payoff of [indx]
    def calculate_payoff(self, indx):
        payoff = 0
        [reals, fakes, factchecks] = self.count_strategy_neighbors(indx)
        
        # if [indx] is real
        if self.players[indx].real:
            payoff += reals * self.parameters.payoff[0]
            payoff += fakes * self.parameters.payoff[1]
            if self.parameters.accuracy == 1:
                payoff += factchecks * self.parameters.payoff[2]
            else:
                for i in range(factchecks):
                    r = rand.random()
                    if r <= self.parameters.accuracy:
                        payoff += self.parameters.payoff[2]
                    else:
                        payoff += self.parameters.payoff[5]
        
        # if [indx] is fake                
        elif self.players[indx].fake:
            payoff += reals * self.parameters.payoff[3]
            payoff += fakes * self.parameters.payoff[4]
            if self.parameters.accuracy == 1:
                payoff += factchecks * self.parameters.payoff[5]
            else:
                for i in range(factchecks):
                    r = rand.random()
                    if r <= self.parameters.accuracy:
                        payoff += self.parameters.payoff[5]
                    else:
                        payoff += self.parameters.payoff[2]
       
        # test if [indx] is not a fact-checker                
        else:
            if not self.players[indx].factcheck:
                print("Payoff Calculation Error")
                
        return payoff
    
    
# Updates the strategy of each individual
# Individuals choose to replicate a strategy proportionaly to 
# fitness from average payoff
    def update_step(self):
        #Reset all players .new
        for player in self.players:
            player.new = False
            
        #Create a temporary list to update from
        #True indicates update to real
        update_list = [True]*self.popsize
        
        #Calculate each player's payoff
        payoffs = [0]*self.popsize
        for indx in range(self.popsize):
            if self.degree[indx] != 0:
                payoffs[indx] = self.calculate_payoff(indx)/self.degree[indx]
        
        #Update each player
        for indx in range(self.popsize):
            if not self.players[indx].factcheck:
                #Get list of all non-fact-checker neighbors
                neighbors = []
                for nindx in self.adjlist[indx]:
                    if not self.players[nindx].factcheck:
                        neighbors.append(nindx)
                
                #Get the total fitness of neighbors
                totalfitness = 0
                cumulative_fit = []
                for nindx in neighbors:
                    pipay = self.parameters.selection * payoffs[nindx]
                    totalfitness += math.exp(pipay)
                    cumulative_fit.append(totalfitness)
                    
                #Select neighbor to copy
                choice = -1
                r = rand.random() * totalfitness
                for i in range(len(neighbors)):
                    if cumulative_fit[i]>r:
                        choice = i
                        break
                            
                #Error Check
                if choice == -1 and neighbors:
                    print("Update Error")
                    print(indx)
                    print(neighbors)
                    print(totalfitness)
                    print(cumulative_fit)
                    print(r)
                    
                #Find neighbor and strategy
                if neighbors:
                    chosen_neighbor = neighbors[choice]
                else:
                    chosen_neighbor = indx
                    
                update_list[indx] = self.players[chosen_neighbor].real
                
        #Update player strategies with update_list
        for indx in range(self.popsize):
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
#Presets for the population
#########################################        
    
#Set all the players to real
    def preset_all_real(self):
        for player in self.players:
            player.set_real()
            player.new = False
            
#Set all players to fake
    def preset_all_fake(self):
        for player in self.players:
            player.set_fake()
            player.new = False
            
#Set each player to real or fake with probability 1/2
    def preset_random(self):
        for player in self.players:
            r = rand.random()
            if r<=0.5:
                player.set_real()
            else:
                player.set_fake()
            player.new = False
    
#Randomly select n individuals to become fact-checkers
    def add_n_factcheckers(self, n):
        #List of indices to become fact-checkers
        factcheckers = []
        while len(factcheckers)<n:
            indx = rand.randint(0,self.popsize-1)
            if not indx in factcheckers:
                if not self.players[indx].factcheck:
                    factcheckers.append(indx)
                    
        for indx in factcheckers:
            self.players[indx].set_factcheck()
                
#Randomly select n individuals to become real news
    def add_n_reals(self, n):
        #List of indices to become real
        reals = []
        while len(reals)<n:
            indx = rand.randint(0,self.popsize-1)
            if not indx in reals:
                if not self.players[indx].real:
                    reals.append(indx)
                    
        for indx in reals:
            self.players[indx].set_real()
    
#Randomly select n individuals to become fake news
    def add_n_fakes(self, n):
        #List of indices to become fakes
        fakes = []
        while len(fakes)<n:
            indx = rand.randint(0,self.popsize-1)
            if not indx in fakes:
                if not self.players[indx].fake:
                    fakes.append(indx)
                    
        for indx in fakes:
            self.players[indx].set_fake()
            
#Takes a list of indicies and sets each of those players to fact-checker
    def add_list_facts(self, indxs):
        for indx in indxs:
            self.players[indx].set_factcheck()
                    
#Makes specific players fact-checkers with given probability p
    def add_list_facts_imperfect(self, indxs, p):
        for indx in indxs:
            if rand.random()<p:
                self.players[indx].set_factcheck()
        
#########################################
#Data collection for the population
#########################################        
                
#Count how many members of the population are sharing real news
    def count_reals(self):
        count = 0
        for indx in range(self.popsize):
            if self.players[indx].real:
                count += 1
        return count
    
#Count how many members of the population are sharing fake news
    def count_fakes(self):
        count = 0
        for indx in range(self.popsize):
            if self.players[indx].fake:
                count += 1
        return count
    
#Count how many members of the population are fact-checking
    def count_factchecks(self):
        count = 0
        for indx in range(self.popsize):
            if self.players[indx].factcheck:
                count += 1
        return count

#Count how many member of the population are playing each strategy
    def strategy_count(self):
        count_reals = 0
        count_fakes = 0
        count_factchecks = 0
        for indx in range(self.popsize):
            if self.players[indx].real:
                count_reals += 1
            elif self.players[indx].fake:
                count_fakes += 1
            elif self.players[indx].factcheck:
                count_factchecks += 1
            else:
                print("Strategy Counter Error")
        return [count_reals, count_fakes, count_factchecks]
    
#Create a list with True in every index that is sharing real news
    def reals_list(self):
        reals_list = [False]*self.popsize
        for indx in range(self.popsize):
            if self.players[indx].real:
                reals_list[indx] = True
        return reals_list
    
#Computes the probability that two individuals have the same strategy 
#Based on the number of common neighbors they have
#Does not include fact-checkers, but fcs do count as neighbors
    def neighbor_strat_probs(self):
        same = [0]*max(self.degree)
        dif = [0]*max(self.degree)
        totals = [0]*max(self.degree)
        probs = [0]*max(self.degree)
        
        for indx in range(self.popsize):
            if not self.players[indx].factcheck:
                for nindx in range(self.popsize):
                    if not self.players[nindx].factcheck and indx != nindx:
                        cns = self.common_neighbors(indx, nindx)
                        if self.players[indx].real==self.players[nindx].real:
                            same[cns] += 1
                        else:
                            dif[cns] += 1
            
        for cns in range(self.degree[0]):
            totals[cns] = same[cns]+dif[cns]
            if totals[cns] != 0:
                probs[cns] = same[cns]/totals[cns]
    
        return probs, totals
    
    def pair_probabilities(self):
        same = 0
        dif = 0
        
        for indx in range(self.popsize):
            if not self.players[indx].factcheck:
                for nindx in range(self.popsize):
                    if not self.players[nindx].factcheck and indx != nindx:
                        if self.players[indx].real == self.players[nindx].real:
                            same += 1
                        else:
                            dif += 1
        
        prob = same/(same+dif)
        
        return prob
                        
    
    
#########################################
#Network Statistics
#########################################      
    
#Performs a breadth-first search on the subgraph of real news players
#Returns a list with the sizes of the components
    def real_components(self):
        sizes = []
        #The list of indices of players sharing real news
        reals = []
        for indx in range(self.popsize):
            if self.players[indx].real:
                reals.append(indx)
                
        #Breadth-first search
        while reals:
            component = [reals[0]]
            reals.pop(0)
            pointer = 0
            while pointer < len(component):
                for indx in self.adjlist[component[pointer]]:
                    if indx in reals:
                        component.append(indx)
                        reals.pop(reals.index(indx))
                pointer += 1
            sizes.append(len(component))
            
        sizes.sort(reverse = True)
        return sizes
                        
#Returns the eigenvalue centrality of each node
    def cent_eig(self):
        
        if self.popsize>400000:
            eval, evecs = eigsh(self.sparseadjmat, k=1, which='LM')
            
            centrality = evecs
            
            
        else:
            evals, evecs = np.linalg.eigh(self.adjmat)
            
            #Identify if the largest eigenvalue is the first or last one
            e1 = evals[0]
            e2 = evals[self.popsize-1]
            if abs(e2) >= abs(e1):
                indx = self.popsize-1
            else:
                indx = 0
                
            centrality = evecs[:,indx]
                
        #Multiply the eigenvector by -1 if necessary
        if centrality[0] < 0:
            centrality = centrality * -1
       
        return centrality
    
#Returns the betweenness centrality of each node
    def cent_between(self):
        centrality = np.asarray([0]*self.popsize)
        #Find all the geodesics starting at each node
        for indx in range(self.popsize):
            print(indx)
            #List that will tell how many geodesics start at indx
            geos = np.asarray([0]*self.popsize)
                
            #Find distances and weights
            distances = np.asarray([-1]*self.popsize)
            weights = np.asarray([-1]*self.popsize)
            dist = 0
            distances[indx] = 0
            weights[indx] = 1
            
            while dist in distances:
                templist = np.where(distances == dist)[0]
                for nindx in templist:
                    for mindx in self.adjlist[nindx]:
                        if distances[mindx] == -1:
                            distances[mindx] = dist + 1
                            weights[mindx] = weights[nindx]
                        elif distances[mindx] == dist + 1:
                            weights[mindx] += weights[nindx]
                dist += 1
                
            self.dist = distances
            
            #Assign scores
            #Start at the bottom of the tree, whose distance is dist
            #Dont calculate geos through indx here because they will be 
            #double counted
            print('done1')
            while dist > 0:    
                print(dist)
                current = np.where(distances == dist)[0]
                farther = np.where(distances == dist+1)[0]
                for nindx in current:
                    geos[nindx] = 1
                    temp = list(set(self.adjlist[nindx]).intersection(farther))
                    for mindx in temp:
                        if geos[mindx] == -1:
                            print('betweenness error')
                        geos[nindx]+=geos[mindx]*weights[nindx]/weights[mindx]
            
                dist -= 1
                
            #Add the geodesic from indx to itself    
            geos[indx] = 1
            
#            print(f'index {indx}')
#            print(geos)
                
            centrality += geos                
                
        return centrality
        
#Returns the local clustering coefficient    
    def clusteringcoeff(self, indx):
        coefficient = 0
        indxs = self.adjlist[indx]
        for nindx in indxs:
            coefficient += len(set(indxs) & set(self.adjlist[nindx]))/2
            
        if len(indxs) > 1:
            coefficient = coefficient / comb(len(indxs), 2)
        return coefficient
            
#Returns the number of neighbors in common between indx and nindx
    def common_neighbors(self, indx, nindx):
        neighbors = len(set(self.adjlist[indx]) & set(self.adjlist[nindx]))
        
        return neighbors
        
    
    
    