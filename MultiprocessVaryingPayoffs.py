########################################################
# Creates multiple random populations and plays the game for a given number of
# time steps or until fixation is achieved
# Uses multiprocessing to utilize multiple cores simultaneously
# Copyright (C) 2023  Matthew Jones
########################################################

import multiprocessing
from PopulationClass import Population
import matplotlib.pyplot as plt


def run_simulations_sw(params):
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
    bb_payoff = params[3]

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
        pop.pop_params.payoff = [1.0, 0.0, 1.0, 0.0, bb_payoff, -4.0, 0.0, 0.0, 0.0]

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
    return results


def run_simulations_grid(params):
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
    # n = 900
    # c = 4
    # p = 0

    # Scale Free Population
    #    n = 900
    #    c = 5

    # Grid Population
    m1 = 30
    m2 = 30
    n = m1 * m2
    k = 8
    p = 0

    # All Population Structures
    num_of_pops = params[1]
    maxtime = params[2]
    psdetection = 20
    bb_payoff = params[3]

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

        pop = Population(
            "grid", grid_width=m1, grid_height=m2, pop_size=n, node_degree=k, p=p
        )
        pop.pop_params.payoff = [1.0, 0.0, 1.0, 0.0, bb_payoff, -4.0, 0.0, 0.0, 0.0]

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
    return results


if __name__ == "__main__":

    num_of_pops = 1
    maxtime = 5000

    poolsize = 20

    pc_step = 0.025
    iters = 200

    # Remember to change this as necessary
    print("Population structure is small world network: n = 900, c=4, p=0.0")
    print(f"{iters} populations for each density")
    print(f"Max time is {maxtime} time steps")

    bb_payoffs = [1 + 0.1 * i for i in range(21)]

    crit_fracs_sw = []
    print("Beginning Small World")
    pc = 0
    for bb_payoff in bb_payoffs:
        print(bb_payoff)

        real_prob = 0
        # pc = 0
        old_prob = 0

        while real_prob < 0.5:
            print(pc)
            params = [[pc, num_of_pops, maxtime, bb_payoff] for _ in range(iters)]
            pool = multiprocessing.Pool(processes=poolsize)
            result = pool.map(run_simulations_sw, params)
            real_advs = len([x for x in result if x[0] == 1 or x[2] == 1])
            if real_advs > iters / 2:
                crit_frac = pc - pc_step * (real_advs / iters - 0.5) / (
                    real_advs / iters - old_prob
                )
                pc -= pc_step
                break
            else:
                pc += pc_step
                old_prob = real_advs / iters

        print(f"Critical p_c value is {crit_frac}")
        if crit_frac > 0:
            crit_fracs_sw.append(crit_frac)
        else:
            crit_fracs_sw.append(0)

    crit_fracs_grid = []
    print("Beginning Grid")
    pc = 0
    for bb_payoff in bb_payoffs:
        print(bb_payoff)

        real_prob = 0
        # pc = 0
        old_prob = 0

        while real_prob < 0.5:
            print(pc)
            params = [[pc, num_of_pops, maxtime, bb_payoff] for _ in range(iters)]
            pool = multiprocessing.Pool(processes=poolsize)
            result = pool.map(run_simulations_grid, params)
            real_advs = len([x for x in result if x[0] == 1 or x[2] == 1])
            if real_advs > iters / 2:
                crit_frac = pc - pc_step * (real_advs / iters - 0.5) / (
                    real_advs / iters - old_prob
                )
                pc -= pc_step
                break
            else:
                pc += pc_step
                old_prob = real_advs / iters

        print(f"Critical p_c value is {crit_frac}")
        if crit_frac > 0:
            crit_fracs_grid.append(crit_frac)
        else:
            crit_fracs_grid.append(0)

    SMALL_SIZE = 14
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 22

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.plot(bb_payoffs, crit_fracs_sw)
    plt.plot(bb_payoffs, crit_fracs_grid)
    plt.plot(bb_payoffs, [(x - 1) / (9 + x) for x in bb_payoffs])
    plt.xlabel("$B-B$ payoff")
    plt.ylabel("Critical $p_C$")
    plt.legend(["Small-world", "Grid", "Well-Mixed"])
    plt.tight_layout()
    plt.savefig("varying_BB_payoff.png", dpi=300)
