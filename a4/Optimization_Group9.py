from pulp import *
import pandas as pd

# increase console output
desired_width = 520
pd.set_option('display.width', desired_width)


def run_opt_model(time_horizon=25, use_consultancy_predictions=True, max_engine_constraint=False, waiting_constraint=False):
    """
    Runs the optimization for all given optimization tasks. By setting parameters it is possible to choose from
    optimization tasks.
    :param time_horizon: t_p the given planning time horizon. Defaults to 25 if no value is given.
    :param use_consultancy_predictions: If true the consultancy predictions are used, otherwise, predictions from
     predictions are used.
    :param max_engine_constraint: If true a max engine constraint is introduced for teams, otherwise, not.
    :param waiting_constraint: If true a waiting time is added to a team's maintenance time, otherwise, not.
    :return: nothing.
    """
    if use_consultancy_predictions:
        data = pd.read_excel('RUL_consultancy_predictions.xlsx')
    else:
        data = pd.read_csv('DataSchedulePredicted.csv')
        data = data.astype(int)
    data.rename(columns={'RUL': 'rul'}, inplace=True)

    if waiting_constraint:
        data['w_a'] = 1
        data['w_b'] = 2
    else:
        data['w_a'] = 0
        data['w_b'] = 0

    print('Prepare values / Pre-compute')
    nr_engines = 100
    teams = {1: 'mu_a', 2: 'mu_a', 3: 'mu_b', 4: 'mu_b'}
    waiting_times = {1: 'w_a', 2: 'w_a', 3: 'w_b', 4: 'w_b'}
    max_engines = {1: 2, 2: 2, 3: 2, 4: 2}

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

    # create variables which are used to loop over
    data = data[data['rul'] <= time_horizon]
    M = data['id'].tolist()
    T = range(1, time_horizon + 1)
    I = range(1, len(teams) + 1)

    # create pulp variables
    c = LpVariable.dicts('c', M, lowBound=0, cat=LpInteger)
    x = LpVariable.dicts('x', [(i, j, t) for i in I for j in M for t in T], lowBound=0, cat=LpBinary)

    print('Creating model')
    # objective function
    model = LpProblem("opt", LpMinimize)
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
                w = data.loc[data['id'] == j, waiting_times[i]].item()

                model += lpSum(x[(i, _j, _t)] for _j in M for _t in range(t, min(time_horizon + 1, t + mu + w))) - 1 \
                         <= time_horizon * (1 - x[(i, j, t)])

    # a team must finish their work within T
    for i in I:
        for j in M:
            for t in T:
                mu = data.loc[data['id'] == j, teams[i]].item()

                model += lpSum((t + mu) * x[(i, j, t)]) <= time_horizon + 1

    # cost constraint
    for j in M:
        available_days = time_horizon - data.loc[data['id'] == j, 'rul'].item()
        cost = data.loc[data['id'] == j, 'cost'].item()

        model += cost * (available_days - lpSum(x[(i, j, t)] *
                                                (time_horizon - t - data.loc[data['id'] == j, teams[i]].item() + 1)
                                                for i in I for t in T)) == c[j]

    # cost is either 0 or higher - non-negativity constraint
    for j in M:
        model += c[j] >= 0

    # adds max engine constraint if max_engine_constraint == True
    if max_engine_constraint:
        for i in I:
            k = max_engines[i]

            model += lpSum(x[(i, j, t)] for j in M for t in T) <= k

    print('Solving model')
    model.solve(pulp.PULP_CBC_CMD(maxSeconds=60))
    print(LpStatus[model.status])

    print("Total cost: {}".format(pulp.value(model.objective)))
    print("Cost per engine and team to engine to time assignment:")
    TOL = 0.00001
    for j in model.variables():
        if j.varValue > TOL:
            if j.name[0] == 'c':
                print(f"Cost engine {j.name} = {str(j.varValue)}")
            if j.name[0] == 'x':
                x_split = j.name.split(',')
                team = x_split[0].split('(')[1]
                engine = x_split[1].replace('_','')
                start_day = x_split[2].replace('_','')[:-1]

                mu = data.loc[data['id'] == int(engine), teams[int(team)]].item()
                w = data.loc[data['id'] == int(engine), waiting_times[int(team)]].item()
                end_day = int(start_day) + mu + w - 1

                print(
                    f'Team {team} maintains engine {engine}. The team begins on day {start_day} and ends on day {str(end_day)}')


print('#####################################')
print('# Optimization Task 1_B             #')
print('#####################################')
run_opt_model(use_consultancy_predictions=False)
print('#####################################')

print('# Optimization Task 1_C             #')
print('#####################################')
run_opt_model()
print('#####################################')

print('# Optimization Task 1_D             #')
print('#####################################')
print('Own predictions: ')
run_opt_model(time_horizon=40, use_consultancy_predictions=False)
print('Consultancy predictions: ')
run_opt_model(time_horizon=40)
print('#####################################')

print('# Optimization Task 2_B             #')
print('#####################################')
run_opt_model(use_consultancy_predictions=False, max_engine_constraint=True)
print('#####################################')

print('# Optimization Task 2_C             #')
print('#####################################')
run_opt_model(max_engine_constraint=True)
print('#####################################')

print('# Optimization Task 3_B             #')
print('#####################################')
run_opt_model(use_consultancy_predictions=False, max_engine_constraint=True, waiting_constraint=True)
print('#####################################')

print('# Optimization Task 3_C             #')
print('#####################################')
run_opt_model(max_engine_constraint=True, waiting_constraint=True)
print('#####################################')
