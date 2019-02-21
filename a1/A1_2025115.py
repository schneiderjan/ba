from pulp import *
import pandas as pd
import numpy as np

df = pd.read_excel('fld.xlsx')

matrix = df.values
# print(matrix)
locs_by_name = list(df.columns)
locations = []
locations.extend(range(1, 38))
capacity = 4000
demand = 100
fix_cost1 = 100000
fix_cost2 = 150000

# create a dictionary of dictionary that contains all costs for any location
# i.e. {1: {1: 0, 2: 160, 3: 3082, 4: 1639, ....
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
# s_i - capacity
s = LpVariable.dicts("s", locations, 0, capacity)
# d_j - demand
d = LpVariable.dicts("d", locations, 0, demand)

f = dict()
d = dict()
s = dict()
for k in locations:
    f[k] = fix_cost1
    d[k] = demand
    s[k] = capacity

# fy = lpSum(y[i]*f[i] for i in locations)
# cx = lpSum(costs[i][j] * x[(i, j)] for i in locations for j in locations)
objective_function = lpSum(y[i]*f[i] for i in locations) + \
                     lpSum(costs[i][j] * x[(i, j)] for i in locations for j in locations)

# objective function
model += lpSum(objective_function)

# constraints
for i in locations:
    model += lpSum(x[(i, j)] for j in locations) == d[i]

for i in locations:
    for j in locations:
        model += lpSum(x[(i, j)]) <= y[j]*s[j]

model.solve()
print(LpStatus[model.status])

TOL = 0.00001
nr_of_dcs = 0
for i in locations:
    if y[i].varValue > TOL:
        nr_of_dcs += 1
        print("DC in", locs_by_name[i-1])
print("Nr. of DC's set up is {}".format(nr_of_dcs))


