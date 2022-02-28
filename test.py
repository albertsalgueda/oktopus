from main import Campaign,State,AI

#we should do the testing here
#i'll start manually but we should automate it

campaign1 = Campaign(1,0,0,0,0,0)
campaign2 = Campaign(2,0,0,0,0,0)
campaign3 = Campaign(3,0,0,0,0,0)

campaigns = [campaign1,campaign2,campaign3]

oktopus = AI()
current_state = State(50,100,campaigns,oktopus)

current_state.allocation()

print(current_state.get_state())

