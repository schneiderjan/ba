import pandas as pd
from math import radians, cos, sin, asin, sqrt
import numpy as np
import random


df = pd.read_excel('Ex2.1-2025115.xls')
data = pd.read_excel('Data Excercise 2 - EMTE stores - BA 2019.xlsx')

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
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def make_distance_matrix(df):
    """
    Extracts location from df
    :param df: The given data set
    :return: A matrix A, the index corresponds to the number of the location and the content is the distance
    btw. the locations
    """
    A = np.zeros(shape=(df.shape[0], df.shape[0]))
    for i, row_i in enumerate(df.itertuples()):
        for j, row_j in enumerate(df.itertuples()):
            # calculate distance between i and j
            A[i][j] = haversine(row_i.Long, row_i.Lat, row_j.Long, row_j.Lat)
    return A

def make_list_containing_lists_routes(routes):
    hq_counter = 0
    temp_route = []
    list_of_lists_of_routes = []
    for city in routes:
        # print(city)
        if city == 0:
            hq_counter += 1
            temp_route.append(city)
            if hq_counter == 2:
                # print(temp_route)
                list_of_lists_of_routes.append(temp_route)
                temp_route = []
                hq_counter = 0
        else:
            temp_route.append(city)

    return list_of_lists_of_routes
# def two_opt_swap(route, i, j):

def are_edges_in_same_route(routes_of_routes, i, j):
    for route in routes_of_routes:
        if i in route and j in route:
            return True
            print("in same route")
        else:
            return False


def two_opt_swap(output_df, data, n_iterations):
    A = make_distance_matrix(data)
    routes = df['City Nr.']
    routes_of_routes = make_list_containing_lists_routes(routes)

    for i in range(n_iterations):
        nodeA = random.randint(1, 133)
        nodeB = routes[nodeA + 1]
        nodeC = random.randint(1, 133)
        nodeD = routes[nodeC + 1]

        # Basically if the km are lower for routes in total the swap can be made
        # but also the whole time needs to be still in the 8 visit and 10 john work time
        if are_edges_in_same_route(routes_of_routes, nodeA, nodeC):
            pass
        else:
            costAD, costBC = check_AD_swap(nodeA, nodeD)
            costAC, costBD = check_AC_swap(nodeA, nodeC)


two_opt_swap(df, data, 100000)

