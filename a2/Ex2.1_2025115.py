from math import radians, cos, sin, asin, sqrt
import pandas as pd
import numpy as np

# See console output
print("Pandas version: {}".format(pd.__version__))

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def calculate_travel_time(distance):
    """
    Calculates the time spent travelling for a specified distance
    :param distance: the distance to be travelled
    :return: time spent travelling
    """
    driving_speed = 80
    return distance / driving_speed

def make_distance_matrix(df):
    """
    Extracts location from df
    :param df: The given data set
    :return: A matrix A, the index corresponds to the number of the location and the content is the distance
    btw. the locations
    """
    print(df.shape[0])
    A = np.zeros(shape=(df.shape[0], df.shape[0]))
    for i, row_i in enumerate(df.itertuples()):
        for j, row_j in enumerate(df.itertuples()):
            # calculate distance between i and j
            A[i][j] = haversine(row_i.Long, row_i.Lat, row_j.Long, row_j.Lat)
    return A

def are_all_locations_visited(df):
    x = df.visited[1:].all()
    # print("x")
    # print(x)

def get_visiting_time(df, index):
    store_type = df.loc[index].Type
    if store_type == "Jumbo":
        return 1.5
    else:
        return 1


def can_next_location_be_visited(current_index, next_index, time_travelled, df):
    """
    Determines if the given nr/index of a location can be visited considering the contraints
    on travel time, visiting time, max working hours, etc.
    :param next_index: the next location with minimum travel cost
    :return: True, if location can be visited. False, otherwise.
    """
    daily_travel_time = 10
    daily_visiting_time = 17 - 9

    distance_to_travel = haversine(df.loc[current_index].Long, df.loc[current_index].Lat, df.loc[next_index].Long, df.loc[next_index].Lat)
    time_to_travel = calculate_travel_time(distance_to_travel)

    if time_travelled == 0 and time_to_travel < 1:
        time_travelled += time_to_travel
        time_travelled += 1 - time_to_travel # john is waiting for the shop to open
        time_travelled += get_visiting_time(df, next_index)
        df.loc[next_index, 'visited'] = True
    elif time_travelled > 0:
        visiting_time = get_visiting_time(df, next_index)
        if (time_travelled + time_to_travel + visiting_time) <= 9: #john can travel and make visit before 17h
            time_travelled += time_to_travel
            time_travelled += get_visiting_time(df, next_index)
            df.loc[next_index, 'visited'] = True

        # travel time plus visitng is in time frame and he makes the visit
    #     or either travel time or visiting time to long, to be in visiting time frame

    #         two options during the day he can visit
#      or he can / cannot visit
#           because travel time longer than opening or visiting time longer than opening

# last thing he must be able to get home. he must get back to index 0




#     be too early -> add time until shop opens
#     add the time based on the store type
#     check if visit be made in time, so, before 17. otherwise set something in df and clear at the end of the day
#     need to check if john can still make it back if he did the visit


def nearest_neighbor(df, start=0):
    """Nearest neighbor algorithm.
    A is an NxN array indicating distance between N locations
    start is the index of the starting location
    Returns the path and cost of the found solution
    """
    A = make_distance_matrix(df)

    time_travelled = 0
    route_counter = 0
    route_distance = 0
    total_distance = 0

    path = [start]
    N = A.shape[0]
    mask = np.ones(N, dtype=bool)   # boolean values indicating which
                                    # locations have not been visited
    mask[start] = False

    while not are_all_locations_visited(df):
        for i in range(N-1):
            current_index = path[-1]
            next_index = np.argmin(A[current_index][mask]) # find minimum of remaining locations
            # check if visited or constraints are not ok

            if can_next_location_be_visited(current_index, next_index, time_travelled, df):
                next_loc = np.arange(N)[mask][next_index]  # convert to original location
                path.append(next_loc)
                mask[next_loc] = False
                route_distance += A[current_index, next_loc]
            #     set next location
            else:
                continue

    return path, cost

# ex1 nearest neighbour
# EMTE likes to know how many days and how many kilometers John needs in total to visit all the stores.
df = pd.read_excel("Data Excercise 2 - EMTE stores - BA 2019.xlsx")
df['visited'] = False
df['cannot_visit'] = False
df['route_dist'] = 0
df['total_dist'] = 0

print(df.loc[0].Nr)
print(df.loc[0].Name)

path, cost = nearest_neighbor(df)
# when route starts john can travel less than an hour and wait for 9h to start visit.

