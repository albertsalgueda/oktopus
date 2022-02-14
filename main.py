
class Campaign():
    def __init__(self,id,budget,spent,conversions,roi):
        self.id = id    
        self.budget = budget
        self.spent = spent
        self.conversions = conversions
        self.roi = roi


class State(Campaign):

    def __init__(self,budget,time,campaigns):
        self.budget = budget
        self.time = time
        self.campaigns = campaigns
    