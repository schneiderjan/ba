from pulp import *
import pandas as pd

desired_width = 520
pd.set_option('display.width', desired_width)

data = pd.read_excel('RUL_consultancy_predictions.xlsx')
data.rename(columns={'RUL': 'rul'}, inplace=True)
print('Preparing values')
# nr of engines
nr_engines = 100
I_a = 2
I_b = 2
# I = 4
time_horizon = 25
# T = t_p
teams = {1: 'mu_a', 2: 'mu_a', 3: 'mu_b', 4: 'mu_b'}
# mu = {'a': 'mu_a', 'b': 'mu_b'}
# variables
cost = dict()
mu_A = dict()
mu_B = dict()
rul = dict()

data['cost'] = -1
data['mu_a'] = -1
data['mu_b'] = -1

for i in range(nr_engines):
    # cost
    if 0 <= i <= 19 or 60 <= i <= 79:
        data.at[i, 'cost'] = 5
    elif 20 <= i <= 39:
        data.at[i, 'cost'] = 7
    elif 40 <= i <= 59:
        data.at[i, 'cost'] = 9
    elif 80 <= i <= 99:
        data.at[i, 'cost'] = 3

    # mu a
    if 0 <= i <= 24:
        data.at[i, 'mu_a'] = 4
    elif 25 <= i <= 49:
        data.at[i, 'mu_a'] = 6
    elif 50 <= i <= 74:
        data.at[i, 'mu_a'] = 3
    elif 75 <= i <= 99:
        data.at[i, 'mu_a'] = 5

    # mu b
    if 0 <= i <= 32 or 67 <= i <= 99:
        data.at[i, 'mu_b'] = data.at[i, 'mu_a'] + 1
    elif 33 <= i <= 66:
        data.at[i, 'mu_b'] = data.at[i, 'mu_a'] + 2

# make things to loop over starting at 1
data = data[data['rul'] <= time_horizon]
M = data['id'].tolist()
T = range(1, time_horizon + 1)
I = range(1, len(teams) + 1)

c = LpVariable.dicts('c', M, lowBound=0, cat=LpInteger)
x = LpVariable.dicts('x', [(i, j, t) for i in I for j in M for t in T], lowBound=0, cat=LpBinary)

print('Creating model')

# model
model = LpProblem("opt1_B", LpMinimize)
objective_function = lpSum(c[j] for j in M)
model += objective_function

print('Adding constraints')

# constraints
# team to engine
for j in M:
    model += lpSum(x[(i, j, t)] for i in I for t in T) <= 1

# team blocking
for i in I:
    for j in M:
        for t in T:
            mu = data.loc[data['id'] == j, teams[i]].item()
            model += lpSum(x[(i, _j, _t)] - 1 for _j in M for _t in range(t, min(time_horizon + 1, t + mu - 1))) <= \
                     time_horizon * (1 - x[(i, j, t)])

# a team must finish their work within T
for i in I:
    for j in M:
        for t in T:
            mu = data.loc[data['id'] == j, teams[i]].item()

            model += lpSum((t + mu) * x[(i, j, t)]) <= time_horizon + 1

# cost constraint
# test = []
# for i in I:
for j in M:
    available_days = time_horizon - data.loc[data['id'] == j, 'rul'].item()
    # print(available_days)
    # mu = data.loc[data['id'] == j, teams[i]].item()
    cost = data.loc[data['id'] == j, 'cost'].item()
    model += cost * (available_days - lpSum(x[(i, j, t)] * (time_horizon - t - data.loc[data['id'] == j, teams[i]].item() + 1) for i in I for t in T)) == c[j]

# print(test)



# # team blocker - the one that should work but does not.
# for i in I:
#     for t in T:
#         model += lpSum(
#             x[(i, _j, _t)] for _j in M
#                        for _t in range(t, min(time_horizon, t + data.loc[data['id'] == _j, teams[i]].item()))
#                        )-1 <= 1

# cost is either 0 or higher
for j in M:
    model += c[j] >= 0


print('Solving model')
# status = model.solve(pulp.PULP_CBC_CMD(maxSeconds=320))
status = model.solve()
print(LpStatus[model.status])

print("Total cost: {}".format(pulp.value(model.objective)))
TOL = 0.00001
for j in model.variables():
    if j.varValue > 0:
        print(j.name + " = " + str(j.varValue))
