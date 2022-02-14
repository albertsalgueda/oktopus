
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

    def __init__(self,budget,time,campaigns):
        self.budget = budget
        self.time = time
        self.campaigns = campaigns
        self.budget_allocation = {}

    def get_budget_allocation(self):
        #returns a dictionary with the budget allocation
        for campaign in campaigns:

