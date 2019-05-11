
from numpy.random import choice

def norm_choice(options, distribution):
    norm_prob = norm(distribution)
    return choice(options, 1, p=norm_prob)[0]

def norm(v):
    s = sum(v)
    return [i/s for i in v]

r = norm_choice(['a', 'b', 'c'], [0.06, 0.03, 0.25])
print(r)
