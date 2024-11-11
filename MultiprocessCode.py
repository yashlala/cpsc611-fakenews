########################################################
# Creates multiple random populations and plays the game for a given number of
# time steps or until fixation is achieved
# Uses multiprocessing to utilize multiple cores simultaneously
# Copyright (C) 2023  Matthew Jones
########################################################

import multiprocessing
from PopulationClass import Population


def run_simulations(params):
    ###################################
    # Parameters to adjust
    ###################################
    # Random Population
    #    n = 900
    #    p = 0.01

    # K Regular Population
    #    n = 900
    #    k = 5

    # SW Population
    n = 900
    c = 4
    p = 0

    # Scale Free Population
    #    n = 900
    #    c = 5

    # Grid Population
    #    m1 = 30
    #    m2 = 30
    #    n = m1*m2
    #    k = 8
    #    p = 0

    # All Population Structures
    num_of_pops = params[1]
    maxtime = params[2]
    psdetection = 20

    ###################################
    # Initialize
    ###################################
    fcdensity = params[0]
    factcheckers = fcdensity * n

    real_fixations = 0
    fake_fixations = 0
    real_advantages = 0
    fake_advantages = 0
    real_fix_times = 0
    fake_fix_times = 0

    for population_number in range(num_of_pops):

        pop = Population("smallworld", pop_size=n, c=c, p=p)

        # Create initial strategies
        pop.randomize_all_nodes()
        pop.add_factchecker_nodes(factcheckers)

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
                # print('The real news strategy has completely fixated')
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
                    # print('The real news strategy has more players')
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
                    # print('The real news strategy has more players')
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
                    # print('The real news strategy has more players')
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
    fcpoints = 10
    num_of_pops = 10
    maxtime = 5000

    poolsize = 5

    # Remember to change this as necessary
    print("Population structure is small world network: n = 900, c=4, p=0.0")
    print(f"{num_of_pops} populations for each density")
    print(f"Max time is {maxtime} time steps")
    print(f"There are {fcpoints} density data points from 0 to 1")

    params = [[i / fcpoints] for i in list(range(fcpoints))]
    for param in params:
        param.append(num_of_pops)
        param.append(maxtime)

    pool = multiprocessing.Pool(processes=poolsize)
    result = pool.map(run_simulations, params)
    # print(result)
    probs1 = [p[0] for p in result]
    probs2 = [p[1] for p in result]
    probs3 = [p[2] for p in result]
    probs4 = [p[3] for p in result]
    time1 = [p[4] for p in result]
    time2 = [p[5] for p in result]
    print(f"Real fixation probs are {probs1}")
    print(f"Fake fixation probs are {probs2}")
    print(f"Real advantage probs are {probs3}")
    print(f"Fake advantage probs are {probs4}")
    print(f"Mean real fixation times are {time1}")
    print(f"Mean fake fixation times are {time2}")
