##############################################################
# Contains the various parameters that a population will have
# Copyright (C) 2023  Matthew Jones
##############################################################


class PopulationParameters:
    def __init__(self):
        # The RGB for the colors used to visualize the population and strategies
        self.REAL_COLOR = [0.0, 0.0, 0.8]
        self.FAKE_COLOR = [0.8, 0.0, 0.0]
        self.FACTCHECK_COLOR = [0.0, 0.8, 0.0]
        self.MISINFOR_COLOR = [0.75, 0.15, 0.85]
        self.NEW_REAL_COLOR = [0.0, 0.5, 1.0]
        self.NEW_FAKE_COLOR = [1.0, 0.5, 0.0]
        # The payoff matrix for the game, vectorized by rows
        # 0: Real news with real news interaction
        # 1: Real news with fake news interaction
        # 2: Real news with fact-checker interaction
        # 3: Real news with misinformation interaction
        # 4: Fake news with real news interaction
        # 5: Fake news with fake news interaction
        # 6: Fake news with fact-checker interaction
        # 7: Fake news with misinformation interaction
        # 8: Fact-checker boost to real news nodes
        # 9: Misinformation boost to fake news nodes
        self.payoff = [1.0, 0.0, 1.0, -3.0, 0.0, 1.0, -2.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0]
        self.accuracy = 1.0
        self.selection = 5.0
