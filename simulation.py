from main import *
from mab import SimulationAgent, AI


def budget_printer(state):
    i=0
    time_step = state.current_time
    for campaign in state.budget_allocation:
        print(f'Campaign {campaign} has a {round(state.budget_allocation[campaign]*100,4)}% of the current budget')
        print(f'Last 5 conversions: {state.campaigns[i].conversion_values[time_step - 5:]}')
        i +=1


def dynamic(state):
    #updates campaign purchase value randomly
    for campaign in state.campaigns:
        previous = campaign.conversion_values[-1]
        new_value = previous  
        new_spent = campaign.budget # / time step lenght
        campaign.update(new_spent,new_value)

print('Welcome to the simulation of Oktopus ;)')
print('We created group of 3 init campaigns for you already')

campaign1 = Campaign(0,0,0,20)
campaign2 = Campaign(1,0,0,50)
campaign3 = Campaign(2,0,0,100)
campaigns = [campaign1,campaign2,campaign3]

budget = int(input('Introduce total budget: '))
time = int(input('Introduce the number of time steps: '))
#inital_allocation = [0.25,0.25,0.5]
state = State(budget,time,campaigns)
optimistic_agent = SimulationAgent(state,1,5)

while state.remaining_budget > 0:
    print('##########################################################################################')
    print(f'IN TIME STEP {state.current_time}:')
    budget_printer(state)
    ## /next
    action = optimistic_agent.act()
    dynamic(state)
    print(f'Remaining budget: {state.remaining_budget}')

print('Optimization finished, thanks for trusting Oktopus')