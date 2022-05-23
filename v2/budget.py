from dataclasses import dataclass
from utils import derivative, fit, data_generator


@dataclass
class Campaign():
    def __init__(self,id,budget):
        self.id = id
        self.budget = budget    # Daily budget vs. Total budget
        self.data_path = str(id) + '.csv'
        data_generator(self.data_path, 1000, 2)
        self.model = self.get_curve()

    def get_curve(self):
        """
        Creates and stores the model for the cost-curve of the campaign. 
        """
        return fit(self.data_path)

    def update(self,spent,conversion_value):
        """
        Updates new information, recomputes the model. 
        """
        pass

class State():
    def __init__(self,id,budget,time,campaigns):
        self.id = id
        self.budget = budget
        self.time = time
        self.campaigns = campaigns
        self.campaign_dict = self.initialize(campaigns)
        self.current_time = 0
        self.current_budget = budget/time

    def initialize(self,campaigns):
        """
        Turn campaigns into a dictionary with key: id, value: campaign object
        """
        k = {}
        for campaign in campaigns:
            k[campaign.id] = campaign
        return k 

    def next(self):
        #   update campaign data
        #   recalculate curves
        #   new budget distribution
        #   allocate budget
        #   recalulate spent & move time
        pass
    
    def distribute(self):
        "given some campaign curves, returns optimal budget allocation "
        budget = self.current_budget
        derivatives = {}
        while budget>0:
            for campaign in self.campaigns:
                derivatives[campaign.id] = derivative(campaign.model,campaign.budget)
            Keymax = max(zip(derivatives.values(), derivatives.keys()))[1]
            self.campaign_dict[Keymax].budget += 1 
            budget -= 1
    
    def print_all(self):
        """
        Saves images of all the active cost curves predictions.
        """
        for campaign in self.campaigns:
            print(f'Campaign {campaign.id} has a budget of {campaign.budget}')
