from utils import *
from budget import Campaign, State
import pandas as pd

k = 3
campaigns = [Campaign(i,0) for i in range(k)]
group = State(0,1000,100,campaigns)
group.distribute()
group.print_all()