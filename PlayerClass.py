###############################################
# Class for each individual in a population. 
# Contains information about strategy
# Copyright (C) 2023  Matthew Jones
################################################


class Player:
    def __init__(self):
        # Initial strategy
        self.real = True
        self.fake = False
        self.factcheck = False
        # No players start having just changed strategy
        self.new = False
        
#Use these to change player strategies
    
    def set_real(self):
        self.real = True
        self.fake = False
        self.factcheck = False
        
    def set_fake(self):
        self.real = False
        self.fake = True
        self.factcheck = False
        
    def set_factcheck(self):
        self.real = False
        self.fake = False
        self.factcheck = True
        
        
        