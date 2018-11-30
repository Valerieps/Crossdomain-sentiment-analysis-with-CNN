import random

with open('sports/4-treated_class_data.out','r') as source:
    data = [ (random.random(), line) for line in source ]
data.sort()
with open('sports/5-shuffled_class_data.out','w') as target:
    for _, line in data:
        target.write( line )