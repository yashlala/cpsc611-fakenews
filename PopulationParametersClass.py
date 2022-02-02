'''Description'''
# Contains the various parameters that a population will have

class PopulationParameters:
    def __init__(self):
#The RGB for the colors used to visualize the population and strategies        
        self.REAL_COLOR = [0.0, 0.0, 0.8]
        self.FAKE_COLOR = [0.8, 0.0, 0.0]
        self.FACTCHECK_COLOR = [0.0, 0.8, 0.0]
        self.NEW_REAL_COLOR = [0.0, 0.5, 1.0]
        self.NEW_FAKE_COLOR = [1.0, 0.5, 0.0]
#The payoff matrix for the game, vectorized by rows        
        self.payoff = [1.0, 0.0, 1.0, 0.0, 2.0, -4.0, 0.0, 0.0, 0.0]
        self.accuracy = 1.0
        self.selection = 5.0
        