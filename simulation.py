from main import *
from mab import SimulationAgent

def budget_printer(campaign_group):
    i=0
    budget = campaign_group.budget_allocation
    for campaign in budget:
        print(f'{campaign} has a {round(budget[campaign]*100,4)}% of the current budget, which is {campaign_group.campaigns[i].budget}')
        print(f'{campaign} has {campaign_group.campaigns[i].conversion_value} conversions')
        i +=1

def dynamic(campaign_group):
    #updates campaign purchase value randomly
    for campaign in campaign_group.campaigns:
        previous = campaign.conversion_value[-1]
        new_value = previous + 1 
        new_spent = campaign.budget # / time step lenght
        campaign.update(new_spent,new_value)

print('Welcome to the simulation of Oktopus ;)')
print('We created group of 3 init campaigns for you already')

campaign1 = Campaign(0,0,0,20)
campaign2 = Campaign(1,0,0,50)
campaign3 = Campaign(2,0,0,100)
campaigns = [campaign1,campaign2,campaign3]

budget = int(input('Introduce total budget: '))
time = int(input('Introduce the number of timestamps: '))
#inital_allocation = [0.25,0.25,0.5]
campaign_group = State(budget,time,campaigns)
optimistic_agent = AI(campaign_group,0.5)

while campaign_group.remaining > 0:
    print('#############################################')
    print(f'Budget at timestamp {campaign_group.current_time} is {campaign_group.current_budget}')
    budget_printer(campaign_group)
    ## /next
    action = optimistic_agent.act()
    dynamic(campaign_group)
    i = input("Ready for the /next time step?")
    print(f'Remaining budget: {campaign_group.remaining}')

print('Optimization finished, thanks for trusting Oktopus')