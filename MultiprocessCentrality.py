########################################################
# Creates multiple random populations, adds fact-checkers according to cent
# plays the game for a given number of
# time steps or until fixation is achieved
# Uses multiprocessing to utilize multiple cores simultaneously
# Copyright (C) 2023  Matthew Jones
########################################################

import multiprocessing
import matplotlib.pyplot as plt
import random as rand
from PopulationClass import Population
import json


def run_simulations(params):
    ###################################
    # Parameters to adjust
    ###################################
    # Random Population
    # n = 1000
    # p = 0.01

    # K Regular Population
    # n = 200
    # k = 5

    # SW Population
    # n = 900
    # c = 8
    # p = 0.03

    # Scale Free Population
    # n = 900
    # c = 5

    # Grid Population
    # m1 = 30
    # m2 = 30
    # n = m1*m2
    # k = 8
    # p = 0

    # Twitter
    n = 404719

    # All Population Structures
    num_of_pops = params[1]
    maxtime = params[2]
    fcprob = params[3]
    psdetection = 20

    ###################################
    # Initialize
    ###################################
    fcdensity = params[0]
    factcheckers = int(fcdensity * n)

    real_fixations = 0
    fake_fixations = 0
    real_advantages = 0
    fake_advantages = 0
    real_fix_times = 0
    fake_fix_times = 0

    for population_number in range(num_of_pops):

        pop = Population("twitter")

        # Create initial strategies
        pop.randomize_all_nodes()
        # Select the individuals to become fact-checkers
        fclist = []

        if params[4] == "Degree":
            cent = list(pop.degree)
        elif params[4] == "Eigenvector":
            cent = list(pop.cent_eig())
        elif params[4] == "Betweenness":
            with open("twitter_betweenness_centrality.json") as f:
                cent = json.load(f)
        elif params[4] == "Random":
            cent = [rand.random() for _ in range(pop.pop_size)]

        for i in range(pop.pop_size):
            c = cent[i]
            cent[i] = [c, i]

        modcent = sorted(cent, reverse=True)

        while len(fclist) < factcheckers:
            val = modcent[0][0]
            tempList = []
            tempList.append(modcent[0][1])
            del modcent[0]

            while modcent and modcent[0][0] == val:
                tempList.append(modcent[0][1])
                del modcent[0]

            if len(tempList) < factcheckers - len(fclist):
                fclist = fclist + tempList
            else:
                addNum = factcheckers - len(fclist)
                fclist = fclist + rand.sample(tempList, addNum)

        pop.set_factchecker_nodes_randomly(fclist, fcprob)

        ###################################
        # Run the simulation
        ###################################

        # Initialization
        oldlist = [True] * pop.pop_size
        olderlist = [True] * pop.pop_size
        newlist = pop.get_realnews_list()
        t = 0
        steady = False
        count = 0
        periodic_count = 0

        # Run the simulation to a steady state
        while not steady:
            t += 1
            #       print(f'time = {t}')

            pop.update_step()

            olderlist = oldlist
            oldlist = newlist
            newlist = pop.get_realnews_list()
            reals = pop.get_total_realnews_count()

            # Detect if a strategy has completely fixated
            if reals == pop.pop_size - factcheckers:
                # print('Thepop_sizeews strategy has completely fixated')
                real_fixations += 1
                real_fix_times += t
                steady = True

            if reals == 0:
                # print('The fake news strategy has completely fixated')
                fake_fixations += 1
                fake_fix_times += t
                steady = True

            # Detect if the system has reached a fixed state, determine the larger strategy
            if oldlist == newlist:
                count += 1
            else:
                count = 0

            if count == psdetection:
                t -= psdetection
                # print('The system has reached a fixed state')
                if reals >= (pop.pop_size - factcheckers) / 2:
                    # print('The pop_sizews strategy has more players')
                    real_fixations += 1
                    real_fix_times += t
                else:
                    # print('The fake news strategy has more players')
                    fake_fixations += 1
                    fake_fix_times += t
                steady = True

            # Detect if the system is in a periodic loop
            if olderlist == newlist:
                periodic_count += 1
            else:
                periodic_count = 0

            if periodic_count == psdetection:
                t -= psdetection
                # print('The system has reached a periodic loop')
                pop.update_step()
                reals += pop.get_total_realnews_count()
                if reals >= pop.pop_size - factcheckers:
                    # print('Thepop_sizeews strategy has more players')
                    real_fixations += 1
                    real_fix_times += t
                else:
                    # print('The fake news strategy has more players')
                    fake_fixations += 1
                    fake_fix_times += t
                steady = True

            # If we reach the time limit:
            if t == maxtime:
                # print('The system has not reached a fixed state')
                if reals >= (pop.pop_size - factcheckers) / 2:
                    # print('The pop_sizews strategy has more players')
                    real_advantages += 1
                else:
                    # print('The fake news strategy has more players')
                    fake_advantages += 1
                steady = True

    # Normalize the results
    if real_fixations != 0:
        real_fix_times = real_fix_times / real_fixations
    if fake_fixations != 0:
        fake_fix_times = fake_fix_times / fake_fixations

    real_fixations = real_fixations / num_of_pops
    fake_fixations = fake_fixations / num_of_pops
    real_advantages = real_advantages / num_of_pops
    fake_advantages = fake_advantages / num_of_pops

    results = [real_fixations, fake_fixations]
    results += [real_advantages, fake_advantages]
    results += [real_fix_times, fake_fix_times]
    return results


if __name__ == "__main__":
    fcpoints = 20
    # num_of_pops = 50
    num_of_pops = 15
    # maxtime = 5000
    maxtime = 200
    fcprob = 1.0

    poolsize = 20

    params = [[i / (fcpoints)] for i in list(range(fcpoints))]
    # params = [0.05, 0.1, 0.2, 0.5, 0.75]
    for param in params:
        param.append(num_of_pops)
        param.append(maxtime)
        param.append(fcprob)
        param.append("Degree")

    fcs = [param[0] for param in params]

    # Remember to change this as necessary
    print("Population structure is the twitter network")
    print(f"{num_of_pops} populations for each density")
    print(f"Max time is {maxtime} time steps")
    print(f"Prob of adding a particular fact-checker is {fcprob}")
    print(f"There are {fcpoints} density points from {fcs[0]} to {fcs[-1]}")

    pool = multiprocessing.Pool(processes=poolsize)
    result1 = pool.map(run_simulations, params)
    print("Degree Done")

    for param in params:
        param[4] = "Eigenvector"
    result2 = pool.map(run_simulations, params)
    print("Eigenvector Done")

    for param in params:
        param[4] = "Betweenness"
    result3 = pool.map(run_simulations, params)
    print("Betweenness Done")

    for param in params:
        param[4] = "Random"
    result4 = pool.map(run_simulations, params)
    print("Random Done")

    degfix = [p[0] for p in result1]
    eigfix = [p[0] for p in result2]
    betfix = [p[0] for p in result3]
    randfix = [p[0] for p in result4]

    print(f"Fact-checker densities are {fcs}")
    print(f"Degree fixation probs are {degfix}")
    print(f"Eigenvector fixation probs are {eigfix}")
    print(f"Betweenness fixation probs are {betfix}")
    print(f"Random fixation probs are {randfix}")

    #########################################
    ## Process Results
    #########################################

    degBFix = [p[1] for p in result1]
    degBAdv = [p[3] for p in result1]
    degAAdv = [p[2] for p in result1]
    degAFix = [p[0] for p in result1]

    eigBFix = [p[1] for p in result2]
    eigBAdv = [p[3] for p in result2]
    eigAAdv = [p[2] for p in result2]
    eigAFix = [p[0] for p in result2]

    betBFix = [p[1] for p in result3]
    betBAdv = [p[3] for p in result3]
    betAAdv = [p[2] for p in result3]
    betAFix = [p[0] for p in result3]

    randBFix = [p[1] for p in result4]
    randBAdv = [p[3] for p in result4]
    randAAdv = [p[2] for p in result4]
    randAFix = [p[0] for p in result4]

    plt.subplots()

    plt.plot(fcs, degBFix, "b")

    plt.plot(fcs, eigBFix, "g")

    plt.plot(fcs, betBFix, "r")

    plt.plot(fcs, randBFix, "c")

    plt.xlabel("$p_C$")
    plt.ylabel("Probability")
    plt.legend(
        ["Degree", "Eigenvector", "Betweenness"],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
    )

    plt.plot(fcs, degBAdv, "--b")
    plt.plot(fcs, degAAdv, "-.b")
    plt.plot(fcs, degAFix, ":b")

    plt.plot(fcs, eigBAdv, "--g")
    plt.plot(fcs, eigAAdv, "-.g")
    plt.plot(fcs, eigAFix, ":g")

    plt.plot(fcs, betBAdv, "--r")
    plt.plot(fcs, betAAdv, "-.r")
    plt.plot(fcs, betAFix, ":r")

    plt.plot(fcs, randBAdv, "--c")
    plt.plot(fcs, randAAdv, "-.c")
    plt.plot(fcs, randAFix, ":c")

    plt.tight_layout()
    plt.savefig("centralityProbs.png", dpi=600)

    bigResults = [result1, result2, result3, result4]

    with open("twitterResults.json", "w") as f:
        json.dump(bigResults, f)
