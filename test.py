from main import Campaign,State

#we should do the testing here
#i'll start manually but we should automate it

campaign1 = Campaign(1,0,0,0,0,0)
campaign2 = Campaign(2,0,0,0,0,0)

campaigns = [campaign1,campaign2]

current_state = State(50,100,campaigns)

a= {"1": 0.5,"2":0.5}

print(current_state.available_actions(a, 0.01,0.05))

campaign1.change_budget(10)
print(campaign1.budget)