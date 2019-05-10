from pulp import *
import pandas as pd
import numpy as np

data = pd.read_excel('RUL_consultancy_predictions.xlsx')

print('Preparing values')
# nr of engines
M = 100
I_a = 2
I_b = 2
I = 4
T = 25
# T = t_p
teams = {0: 'a', 1: 'a', 2: 'b', 3: 'b'}

# variables
cost = dict()
mu_A = dict()
mu_B = dict()
rul = dict()

for i in range(M):
    if 0 <= i <= 19 or 60 <= i <= 79:
        cost[i] = 5
    elif 20 <= i <= 39:
        cost[i] = 7
    elif 40 <= i <= 59:
        cost[i] = 9
    elif 80 <= i <= 99:
        cost[i] = 3

for i in range(M):
    if 0 <= i <= 24:
        mu_A[i] = 4
    elif 25 <= i <= 49:
        mu_A[i] = 6
    elif 50 <= i <= 74:
        mu_A[i] = 3
    elif 75 <= i <= 99:
        mu_A[i] = 5

for i in range(M):
    if 0 <= i <= 32 or 67 <= i <= 99:
        mu_B[i] = mu_A[i] + 1
    elif 33 <= i <= 66:
        mu_B[i] = mu_A[i] + 2

for i in range(M):
    rul[i] = data.at[i, 'RUL']

mu = {'a': mu_A, 'b': mu_B}

c = LpVariable.dicts("c", range(M), 0)
x = LpVariable.dicts('x', indexs=[(i, j, t) for i in range(I) for j in range(M) for t in range(T)], lowBound=0, upBound=1,
                     cat='binary')
# s = LpVariable.dicts('s', [(i, j, t) for i in range(M) for j in range(M) for t in range(T)], 0, cat='integer')
# t = LpVariable.dicts('t', range(T), lowBound=0, upBound=T, cat='integer')
# f = LpVariable.dicts('f_j', range(M), 0, 1, cat='integer')

print('Creating model')

# model
model = LpProblem("opt1_B", LpMinimize)
objective_function = lpSum(c[j] for j in range(M))
model += objective_function

print('Adding constraints')

# constraints
# team to engine
for j in range(M):
    model += lpSum(x[(i, j, t)] for i in range(I) for t in range(T)) <= 1

# a team must finish their work within T
for i in range(I):
    for j in range(M):
        for t in range(T):
            model += lpSum((t + mu[teams[i]][j]) * x[(i, j, t)]) <= T

# cost constraint
for i in range(I):
    for j in range(M):
        available_days = T - rul[j]
        model += cost[j] * (available_days - lpSum(((x[(i, j, t)]) * (T - t - mu[teams[i]][j])) for t in range(T))) == c[j]

# should be working - solution infeasible
# with more time get negative result and not solved.
# team blocker
for i in range(I):
    for j in range(M):
        for t in range(T):
            model += lpSum(x[(i, w, e)] for w in range(M) for e in range(t, min(T, t + mu[teams[i]][w] - 1)))  \
                      <= T * (1 - x[(i, j, t)])

# # team blocker - instant infeasible
# for i in range(I):
#     for t in range(T):
#         model += lpSum(x[(i, w, e)] for w in range(M) for e in range(max(1, t + mu[teams[i]][w] + 1), T)) <= 1

# cost is either 0 or higher
for j in range(M):
    model += c[j] >= 0

print('Solving model')
status = model.solve(pulp.PULP_CBC_CMD(maxSeconds=120))
# status = model.solve()
print(LpStatus[model.status])

print("Total cost: {}".format(pulp.value(model.objective)))
TOL = 0.00001
for j in model.variables():
    if j.varValue > TOL:
        print(j.name + " = " + str(j.varValue))

# constraints needed:
# [x] engine to team
# [] one engine at at time
# [x] finish work in T
# [x] assign costs
# [] fail only once \\ might be not needed anymore.
# []


# # one team only on one engine at given time.
# for i in range(I):
#     team = teams[i]
#     for j in range(M):
#         for t in range(T):
#             model += lpSum(x[(i, g, h)] for h in range(t, min(T, t + mu[teams[i]][j] - 1)) for g in range(M))  \
#                      <= T - t - rul[j]
#                      # <= M * T * (x[(i, j, t)])

# for i in range(I):
#     for j in range(M):
#         for t in range(T):
#             model += lpSum(x[(i, w, e)] - 1 for w in range(M) for e in range(t, min(T, t + mu[teams[i]][j] - 1))) <= T(1 - x[(i, j, t)])
# team blocker
# for i in range(I):
#     for j in range(M):
#         for t in range(T):
#             model += lpSum(x[(i, k, d)] for d in range(min(T, t + mu[teams[i]][j])) for k in range(M)) <=