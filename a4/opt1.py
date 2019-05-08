from pulp import *
import pandas as pd
import numpy as np

data = pd.read_excel('RUL_consultancy_predictions.xlsx')

print('Preparing values...')
# nr of engines
N = 100
nr_teams_a = 2
nr_teams_b = 2
nr_teams = 4
T = 25
# T = t_p
teams = {0: 'a', 1: 'a', 2: 'b', 3: 'b'}

# variables
cost = dict()
mu_A = dict()
mu_B = dict()
rul = dict()

for i in range(N):
    if 0 <= i <= 19 or 60 <= i <= 79:
        cost[i] = 5
    elif 20 <= i <= 39:
        cost[i] = 7
    elif 40 <= i <= 59:
        cost[i] = 9
    elif 80 <= i <= 99:
        cost[i] = 3

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

mu = {'a': mu_A, 'b': mu_B}

c = LpVariable.dicts("c", range(N), 0, cat='integer')
x = LpVariable.dicts('x', [(i, j, t) for i in range(N) for j in range(N) for t in range(T)], 0, cat='integer')
s = LpVariable.dicts('s', [(i, j, t) for i in range(N) for j in range(N) for t in range(T)], 0, cat='integer')
t = LpVariable.dicts('t', range(T), lowBound=0, upBound=T, cat='integer')
f = LpVariable.dicts('f_j', range(N), 0, 1, cat='integer')

print('Creating model')

# model
model = LpProblem("opt1_B", LpMinimize)
objective_function = lpSum(c[j] for j in range(N))
model += objective_function

print('Adding constraints')

# constraints
# team to engine
for j in range(N):
    for i in range(nr_teams):
        model += lpSum(x[(i, j, t)] for t in range(T)) <= 1

# one team only on one engine at given time.
for i in range(nr_teams):
    team = teams[i]
    for j in range(N):
        for t in range(T):
            model += lpSum(x[(i, g, h)] for h in range(t, min(T, t + mu[teams[i]][j])) for g in range(N)) \
                     <= N * T * x[(i, j, t)]

# a team must finish their work within T
for i in range(nr_teams):
    for j in range(N):
        model += lpSum([mu[teams[i]][j] * x[(i, j, t)] for t in range(T)]) <= T

# cost constraint
for j in range(N):
    available_days = T - rul[j]
    for i in range(nr_teams):
        model += cost[j] * (available_days - lpSum(((x[(i, j, t)]) * (T - t - mu[teams[i]][j])) for t in range(T))) == \
                 c[j]

# cost is either 0 or higher
for j in range(N):
    model += c[j] >= 0

print('Solving model')
status = model.solve(pulp.PULP_CBC_CMD(maxSeconds=600))
# status = model.solve()
print(LpStatus[model.status])

print("Total cost: {}".format(pulp.value(model.objective)))
TOL = 0.00001
nr_of_dcs = 0
for j in model.variables():
    if j.varValue > TOL:
        print(j.name + " = " + str(j.varValue))

# constraints needed:
# [x] engine to team
# [x] one engine at at time
# [x] finish work in T
# [] assign costs
# [] fail only once
# []
