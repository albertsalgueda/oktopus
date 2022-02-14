from main import Campaign,State

#we should do the testing here
#i'll start manually but we should automate it

campaign1 = Campaign(1,0,0,0,0,0)
campaign2 = Campaign(2,0,0,0,0,0)

campaigns = [campaign1,campaign2]

current_state = State(50,100,campaigns)

print(current_state.campaigns)
print(current_state.get_budget_allocation())

campaign1.change_budget(10)
print(campaign1.budget)