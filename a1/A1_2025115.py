from pulp import *
import pandas as pd
import numpy as np

df = pd.read_excel('fld.xlsx')

matrix = df.values
locs_by_name = list(df.columns)
locations = []
locations.extend(range(1, 38))
capacity = 4000
demand = 100
fix_cost = 100000

costs = {}
for i in locations:
    costs[i] = {}

for (x,y), value in np.ndenumerate(matrix):
    key = x+1
    inner_key = y+1
    costs[key][inner_key] = value

model = LpProblem("a1")
# y_i
y = LpVariable.dicts('y', locations, 0, 1, LpBinary)
# x_ij
x = LpVariable.dicts('x',
                        [(i, j) for i in locations
                         for j in locations], 0)
# c_ij
c = LpVariable.dicts('c', costs, 0)
# s_i - capacity
s = LpVariable.dicts("s", locations, 0, capacity)
# d_j - demand
d = LpVariable.dicts("d", locations, 0, demand)

f = dict()
d = dict()
s = dict()
for k in locations:
    f[k] = fix_cost
    d[k] = demand
    s[k] = capacity

fy = lpSum(y[i]*f[i] for i in locations)
cx = lpSum(costs[i][j] * x[(i, j)] for i in locations for j in locations)
objective_function = lpSum(y[i]*f[i] for i in locations) + lpSum(costs[i][j] * x[(i, j)] for i in locations for j in locations)

# objective function
model += lpSum(objective_function)

# constraints
for i in locations:
    model += lpSum(x[(i, j)] for j in locations) == d[i]

for i in locations:
    model += lpSum(x[(i, j)] for j in locations) <= y[i]*s[i]

model.solve()
print(LpStatus[model.status])

TOL = 0.00001
for i in locations:
    if y[i].varValue > TOL:
        print("Location ", i, " produces ", [j for j in locations if x[(i, j)].varValue > TOL])


