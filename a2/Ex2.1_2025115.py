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
    :return: A matrix A the index corresponds to the number of the location and the content is the distance
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
    print("x")
    print(x)



def nearest_neighbor(df , start = 0):
    """Nearest neighbor algorithm.
    A is an NxN array indicating distance between N locations
    start is the index of the starting location
    Returns the path and cost of the found solution
    """
    A = make_distance_matrix(df)
    path = [start]
    cost = 0
    N = A.shape[0]
    mask = np.ones(N, dtype=bool)  # boolean values indicating which
                                    # locations have not been visited
    mask[start] = False

    for i in range(N-1):
        last = path[-1]
        next_ind = np.argmin(A[last][mask]) # find minimum of remaining locations
        next_loc = np.arange(N)[mask][next_ind] # convert to original location
        path.append(next_loc)
        mask[next_loc] = False
        cost += A[last, next_loc]
    are_all_locations_visited(df)

    return path, cost

# ex1 nearest neighbour
# EMTE likes to know how many days and how many kilometers John needs in total to visit all the stores.
df = pd.read_excel("Data Excercise 2 - EMTE stores - BA 2019.xlsx")
df['visited'] = False
df['route_dist'] = 0
df['total_dist'] = 0

path, cost = nearest_neighbor(df)
# when route starts john can travel less than an hour and wait for 9h to start visit.
max_working_hours = 10
visiting_time_jumbo = 1.5
visiting_time_others = 1
# visiting time is 9-17

