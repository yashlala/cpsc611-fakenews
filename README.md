# FakeNews

The following code is used to simulate fake news in a population with citizen
fact-checkers. The section describes the function of different .py files. All
simulation files have many options for different parameters, including network
structure, fact-checker density, and centrality measure for fact-checker
placement.

## Environment Setup

1. Ensure [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is installed on your system.
2. Run `conda env create -f environment.yml` to create a new Conda environment
   with all the required dependencies.
3. Activate the environment with `conda activate cpsc611-fakenews`. 
4. Run the programs you want. Eg: `python3 SinglePopulationRun.py`. 

## Usage Instructions

Multiprocessing requires multiple cores. Because of the size of simulation
necessary for high-quality results, this code was run on a university computing
cluster with many more computing cores than a personal computer. Code running
on 20 cores would typically run in 1-3 days, and a personal computer with fewer
cores will accordingly take longer.

Any of the main .py files (SinglePopulationRun.py, SingleCentrality.py,
MultiprocessCode.py, MultiprocessCentrality.py) should be immediately
executable. Edit these files to adjust the parameters for the simulation as
desired.


## Code Map

- [SinglePopulationRun.py](./SinglePopulationRun.py): Create a single
  population with random fact-checkers
- [SingleCentrality.py](./SingleCentrality.py): Create a single population with
  fact-checkers assigned according to some centrality measure (degree,
  betweenness, etc)
- [MultiprocessCode.py](/MultiprocessCode.py): Create a large ensemble of
  populations with random fact-checkers for simulations
- [MultiprocessCentrality.py](/MultiprocessCentrality.py): Create a large
  ensemble of populations with fact-checkers assigned according to some
  centrality measure for simulations
- [PopulationClass.py](/PopulationClass.py): A class for containing the
  population's state, behavior, etc. Used in all prior files
- [PlayerClass.py](/PlayerClass.py): A class containing the behavior of an
  individual in the simulated population. Used in PopulationClass.py
- [PopulationParameters.py](/PopulationParameters.py): A class
  containing payoff values for the game. Used in PopulationClass.py
- [twitterNetwork.py](/twitterNetwork.py): A file dedicated to computing the
  betweenness centrality of the Twitter network. Input is
  [soc-twitter-follows2.mtx](/soc-twitter-follows2.mtx) and output is
  [twitter_betweenness_centrality](/twitter_betweenness_centrality.json)

## License

This code is covered under the GNU General Public License version 3, available
[here](/LICENSE).



