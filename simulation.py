from main import *

def budget_printer(campaign_group):
    i=0
    budget = campaign_group.allocation()
    for campaign in budget:
        print(f'{campaign} has a {round(budget[campaign]*100)}% of the current budget, which is {campaign_group.campaigns[i].budget}')
        print(f'{campaign} has {campaign_group.campaigns[i].impressions} impressions')
        print(f'{campaign} has {campaign_group.campaigns[i].conversions} conversions')
        print(f'{campaign} has {campaign_group.campaigns[i].roi} ROI')
        i +=1

print('Welcome to the simulation of Oktopus ;)')
print('We created group of 3 init campaigns for you already')

campaign1 = Campaign(1,0,0,0,0,0)
campaign2 = Campaign(2,0,0,0,0,0)
campaign3 = Campaign(3,0,0,0,0,0)
campaigns = [campaign1,campaign2,campaign3]
budget = int(input('Introduce total budget: '))
time = int(input('Introduce the number of timestamps: '))
campaign_group = State(budget,time,campaigns,oktopus)
while campaign_group.remaining > 0:
    print('#############################################')
    print(f'Budget at timestamp {campaign_group.current_time} is {campaign_group.current_budget}')
    budget_printer(campaign_group)
    print(f'Available actions at timestamp {campaign_group.current_time}')
    #actions = list(campaign_group.available_actions(campaign_group.get_state(campaign_group.allocation())))
    #print(actions)
    #action = int(input("Take an action: "))
    action = campaign_group.take_action()
    for campaign in campaign_group.campaigns:
        print(f'Introduce new data for campaign {campaign.id}')
        data = str(input("new Impresions, new Conversions and new ROI respectively separated with a comma: "))
        data = data.split(',')
        campaign.update(data[0],data[1],data[2])
    print(f'Remaining budget: {campaign_group.next_timestamp(action)}')

print('Optimization finished, thanks for trusting Oktopus')