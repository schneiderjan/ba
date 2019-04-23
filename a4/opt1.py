from pulp import *
import pandas as pd
import numpy as np

# nr of engines
N = 100
nr_teams = 2
t_p = 25

#variables
c_j = dict()
mu_A_j = dict()
mu_B_j = dict()

for i in range(N):
    if 0 <= i <= 19 or 60 <= i <= 79:
        c_j[i] = 5
    elif 20 <= i <= 39:
        c_j[i] = 7
    elif 40 <= i <= 59:
        c_j[i] = 9
    elif 80 <= i <= 99:
        c_j[i] = 3

for i in range(N):
    if 0 <= i <= 24:
        mu_A_j[i] = 4
    elif 25 <= i <= 49:
        mu_A_j[i] = 6
    elif 50 <= i <= 74:
        mu_A_j[i] = 3
    elif 75 <= i <= 99:
        mu_A_j[i] = 5

for i in range(N):
    if 0 <= i <= 32 or 67 <= i <= 99:
        mu_B_j[i] = mu_A_j[i] + 1
    elif 33 <= i <= 66:
        mu_B_j[i] = mu_A_j[i] + 2

model = LpProblem("opt1_B", LpMinimize)

c = LpVariable.dicts("c_j", range(N), 0, c_j)
A = LpVariable.dicts("mu_A_j", range(N), 0, mu_A_j)
B = LpVariable.dicts("mu_B_j", range(N), 0, mu_B_j)

