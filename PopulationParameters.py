##############################################################
# Contains the various parameters that a population will have
# Copyright (C) 2023  Matthew Jones
##############################################################


class PopulationParameters:
    def __init__(self, **kwargs):
        # The RGB for the colors used to visualize the population and strategies
        self.REAL_COLOR = [0.0, 0.0, 0.8]
        self.FAKE_COLOR = [0.8, 0.0, 0.0]
        self.FACTCHECK_COLOR = [0.0, 0.8, 0.0]
        self.MISINFOR_COLOR = [0.75, 0.15, 0.85]
        self.NEW_REAL_COLOR = [0.0, 0.5, 1.0]
        self.NEW_FAKE_COLOR = [1.0, 0.5, 0.0]
        # The payoff matrix for the game, vectorized by rows
        self.payoff = [
                kwargs.get('real_from_real', 1.0),       # 0: Real news <- real news interaction
                kwargs.get('real_from_fake', 0.0),       # 1: Real news <- fake news interaction
                kwargs.get('real_from_factcheck', 1.0),  # 2: Real news <- fact-checker interaction
                kwargs.get('real_from_misinfor', -3.0),  # 3: Real news <- misinformation interaction
                kwargs.get('fake_from_real', 0.0),       # 4: Fake news <- real news interaction
                kwargs.get('fake_from_fake', 1.0),       # 5: Fake news <- fake news interaction
                kwargs.get('fake_from_factcheck', -2.0), # 6: Fake news <- fact-checker interaction
                kwargs.get('fake_from_misinfor', 1.0),   # 7: Fake news <- misinformation interaction
                kwargs.get('factcheck_boost', 0.5),      # 8: Fact-checker boost to real news nodes
                kwargs.get('misinfor_boost', 0.5),       # 9: Misinformation boost to fake news nodes
                0.0,
                0.0,
                0.0]
        self.accuracy = 1.0
        self.selection = 5.0
