from utils import derivative

class Campaign():
    pass


class State():
    def __init__(self,budget,time,campaigns):
        self.budget = budget
        self.time = time
        self.campaigns = campaigns
        self.budget_distribution = {}
        self.current_budget = budget/time

    def next(self):
        #   update campaign data
        #   recalculate curves
        #   new budget distribution
        #   allocate budget
        #   recalulate spent & move time
        pass
    
    def distribute(self):
        "given some campaign curves, return optimal budget allocation "
        budget = self.current_budget
        derivatives = {}
        while budget>0:
            for campaign in self.campaigns:
                derivatives[campaign.id] = derivative(campaign.model,campaign.budget)
            Keymax = max(zip(derivatives.values(), derivatives.keys()))[1]
            self.campaigns[Keymax].budget += 1 
            budget -= 1