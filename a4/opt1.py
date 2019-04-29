from pulp import *
import pandas as pd
import numpy as np

data =  pd.read_excel('RUL_consultancy_predictions.xlsx')

# nr of engines
N = 100
nr_teams = 2
danger = 25
# T = t_p

#variables
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
        mu_A[i] = mu_A[i] + 1
    elif 33 <= i <= 66:
        mu_A[i] = mu_A[i] + 2

for i in range(N):
    rul[i] = data.at[i, 'RUL']

c_j = LpVariable.dicts("c_j", range(N), 0, c)
mu_A_j = LpVariable.dicts("mu_A_j", range(N), 0, mu_A)
mu_B_j = LpVariable.dicts("mu_B_j", range(N), 0, mu_B)
x_ij = LpVariable.dicts('x_ij', [(i, j) for i in range(N) for j in range(N)], 0)
t = LpVariable.dicts('t', range(danger), lowBound=0, upBound=danger, cat='integer')
f_j = LpVariable.dicts('f_j', range(N), 0, 1, LpBinary)

#model
model = LpProblem("opt1_B", LpMinimize)

objective_function = lpSum(c_j[i] * max(0, () ) for i in range(N))
