class Campaign():

    def __init__(self,id,budget,spent,impressions,conversions,roi):
        #falta determinar como podemos saber el tiempo que lleva la campaña 
        self.id = id    
        self.budget = budget
        self.spent = spent
        self.impressions = impressions
        self.conversions = conversions
        self.roi = roi

    def change_budget(self,increment):
        #increment debe ser un valor numerico para editar el ( daily budget )
        self.budget = self.budget + increment
    

class State(Campaign):

    def __init__(self,budget,total_time,campaigns):
        self.budget = budget
        self.time = total_time
        self.campaigns = campaigns
        self.current_time = 0
        self.budget_allocation = {}

    def get_timestamp_budget(self):
        #think about it more
        self.current_time +=1
        self.current_budget = self.budget/self.time

    def initial_allocation(self):
        #returns a dict with a proportional allocation
        for campaign in self.campaigns:
            self.budget_allocation[campaign.id] = 1/len(self.campaigns)
        return self.budget_allocation

    def get_state(self):
        #returns a dictionary with the budget allocation
        if self.current_time == 0:
            self.initial_allocation()
        else:
            for campaign in self.campaigns:
                self.budget_allocation[campaign.id] = campaign.budget / self.current_budget
        return self.budget_allocation

    def available_actions(self):
        #returns a set of available actions given a particular state
        pass
    
    @classmethod
    def validate_budget(self,budget_allocation):
        #total budget allocation cannot surpass b 
        #returns True if it's valid
        #returns False if it's not valid
        pass
    
    def allocate_budget(self):
        #changes campaign.budget depending on this timestamp budget and the % budget_allocation
        #calls available actions and takes the best one
        #total budget allocation cannot surpass b 
        pass
