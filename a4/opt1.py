from pulp import *
import pandas as pd
import numpy as np

data = pd.read_excel('RUL_consultancy_predictions.xlsx')

# nr of engines
N = 100
nr_teams_a = 2
nr_teams_b = 2
nr_teams = 4
T = 25
# T = t_p

# variables
c = dict()
mu_A = dict()
mu_B = dict()
rul = dict()

for i in range(N):
    if 0 <= i <= 19 or 60 <= i <= 79:
        c[i] = 5
    elif 20 <= i <= 39:
        c[i] = 7
    elif 40 <= i <= 59:
        c[i] = 9
    elif 80 <= i <= 99:
        c[i] = 3

for i in range(N):
    if 0 <= i <= 24:
        mu_A[i] = 4
    elif 25 <= i <= 49:
        mu_A[i] = 6
    elif 50 <= i <= 74:
        mu_A[i] = 3
    elif 75 <= i <= 99:
        mu_A[i] = 5

for i in range(N):
    if 0 <= i <= 32 or 67 <= i <= 99:
        mu_B[i] = mu_A[i] + 1
    elif 33 <= i <= 66:
        mu_B[i] = mu_A[i] + 2

for i in range(N):
    rul[i] = data.at[i, 'RUL']

# for i in range(N):


print(mu_A)
# print(mu_B)
#
c = LpVariable.dicts("c_j", range(N), 0, c)
mu = LpVariable.dicts("mu", [(i, j) for i in range(nr_teams) for j in range(N)], 0, cat='integer')
team = LpVariable.dicts('team', range(nr_teams), 0, nr_teams, cat='integer')
x = LpVariable.dicts('x', [(i, j, t) for i in range(N) for j in range(N) for t in range(T)], 0, cat='integer')
s = LpVariable.dicts('s', [(i, j, t) for i in range(N) for j in range(N) for t in range(T)], 0, cat='integer')
t = LpVariable.dicts('t', range(T), lowBound=0, upBound=T, cat='integer')
f = LpVariable.dicts('f_j', range(N), 0, 1, cat='integer')
d = LpVariable.dicts('d_j', range(N), lowBound=0, upBound=T, cat='integer')
e = LpVariable.dicts('e_j', range(N), 0, cat='integer')
# model
model = LpProblem("opt1_B", LpMinimize)

# print(x)
# objective function
# objective_function =
model += lpSum(e[j] for j in range(N))

# constraints
for j in range(N):
    for i in range(nr_teams):
        model += lpSum(x[(i, j, t)] for t in range(T)) <= 1

for t in range(T):
    for j in range(N):
        model += lpSum(s[(i, j, t)]) <= 1

print(mu)
for t in range(T):
    for j in range(N):
        model += lpSum(t * s[(i, j, t)] + mu[(i, j)] for i in range(nr_teams)) <= T  # missing:  + mu

for j in range(N):
    model += lpSum(f[j]) <= 1

status = model.solve()
print(LpStatus[model.status])
print(value(model.objective))
# status= model.solve(pulp.PULP_CBC_CMD(maxSeconds=60))
