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


def get_visiting_time(df, index):
    store_type = df.loc[index].Type
    if store_type == "Jumbo":
        return 1.5
    else:
        return 1


def calculate_travel_time(distance):
    """
    Calculates the time spent travelling for a specified distance
    :param distance: the distance to be travelled
    :return: time spent travelling
    """
    driving_speed = 80
    return distance / driving_speed


def make_list_containing_lists_routes(routes):
    hq_counter = 0
    temp_route = []
    list_of_lists_of_routes = []
    for city in routes:
        if city == 0:
            hq_counter += 1
            temp_route.append(city)
            if hq_counter == 2:
                list_of_lists_of_routes.append(temp_route)
                temp_route = []
                hq_counter = 0
        else:
            temp_route.append(city)

    return list_of_lists_of_routes


def are_edges_in_same_route(routes_of_routes, nodeA, nodeC):
    for idx, route in enumerate(routes_of_routes):
        if nodeA in route and nodeC in route:
            return True
    return False


def recalculate_route(A, route, data):
    # print(route)
    total_travel_time = 0
    total_visiting_time = 0
    total_km = 0
    kms = [0]

    current_loc = route[0]
    for idx in range(1, len(route)):
        if idx == 1:
            next_loc = route[idx]
            travel_dist = A[current_loc, next_loc]
            travel_time = calculate_travel_time(travel_dist)
            visiting_time = get_visiting_time(data, next_loc)

            total_travel_time += travel_time
            total_travel_time += visiting_time
            total_visiting_time += visiting_time

            total_km += travel_dist
            kms.append(kms[-1] + travel_dist)

            current_loc = next_loc
        else:
            next_loc = route[idx]
            travel_dist = A[current_loc, next_loc]
            travel_time = calculate_travel_time(travel_dist)
            visiting_time = get_visiting_time(data, next_loc)

            total_travel_time += travel_time
            total_travel_time += visiting_time
            total_visiting_time += travel_time
            total_visiting_time += visiting_time

            total_km += travel_dist
            kms.append(kms[-1] + travel_dist)

            current_loc = next_loc

        if total_visiting_time > 8 or total_travel_time > 10:
            return False, -1, -1, -1

    # print('route good w/ travel time: {}, visit time {}, total km: {}'.format(total_travel_time, total_visiting_time,
    #                                                                          total_km))
    return True, travel_time, total_km, kms


def get_old_routes_total_values(A, route_1, route_2, data):
    is_valid, travel_time_1, total_km_1, kms_1 = recalculate_route(A, route_1, data)
    is_valid, travel_time_2, total_km_2, kms_2 = recalculate_route(A, route_2, data)
    total_km_old = total_km_1 + total_km_2
    total_time_old = travel_time_1 + travel_time_2

    return total_km_old, total_time_old


def check_swap(A, nodeA, nodeB, nodeC, nodeD, output_df, routes_of_routes, data):
    route_1 = []
    route_nr_a = -1
    route_2 = []
    route_nr_c = -1

    for route in routes_of_routes:
        if nodeA in route:
            route_1 = route
            route_nr_a = routes_of_routes.index(route)
        elif nodeC in route:
            route_2 = route
            route_nr_c = routes_of_routes.index(route)

        if len(route_1) > 0 and len(route_2) > 0:
            break

    idx_a = route_1.index(nodeA)
    idx_b = idx_a + 1
    idx_c = route_2.index(nodeC)
    idx_d = idx_c + 1

    # make new route: option 1
    new_route_ad = route_1[:idx_a + 1] + route_2[idx_d:]
    new_route_bc = route_1[:idx_b - 1:-1] + route_2[idx_c::-1]

    # make new route: option 2
    new_route_ac = route_1[:idx_a + 1] + route_2[idx_c::-1]
    new_route_bd = route_1[:idx_b - 1:-1] + route_2[idx_d:]

    is_option_1_valid = False
    is_option_2_valid = False

    # option1
    total_km_opt1 = -1
    total_time_opt1 = -1
    is_route_valid, travel_time_ad, total_km_ad, kms_ad = recalculate_route(A, new_route_ad, data)
    if is_route_valid:
        is_route_valid, travel_time_bc, total_km_bc, kms_bc = recalculate_route(A, new_route_bc, data)
        if is_route_valid:
            is_option_1_valid = True
            total_km_opt1 = total_km_ad + total_km_bc
            total_time_opt1 = travel_time_ad + travel_time_bc

    # option2
    total_km_opt2 = -1
    total_time_opt2 = -1
    is_route_valid, travel_time_ac, total_km_ac, kms_ac = recalculate_route(A, new_route_ac, data)
    if is_route_valid:
        is_route_valid, travel_time_bd, total_km_bd, kms_bd = recalculate_route(A, new_route_bd, data)
        if is_route_valid:
            is_option_2_valid = True
            total_km_opt2 = total_km_ac + total_km_bd
            total_time_opt2 = travel_time_ac + travel_time_bd

    if is_option_1_valid and is_option_2_valid:
        print('both are nice. please compare also to old stuff')
        total_km_old, total_time_old = get_old_routes_total_values(A, route_1, route_2, data)
        if total_km_old <= total_km_opt1 and total_km_old <= total_km_opt2 and total_time_old <= total_km_opt1 and total_time_old <= total_time_opt2:
            print('>>>>>change nothing.')
    elif is_option_1_valid:
        print('only 1st opetion. comapre to old')
        total_km_old, total_time_old = get_old_routes_total_values(A, route_1, route_2, data)
        if total_km_opt1 < total_km_old and total_time_opt1 <= total_time_old:
            print('>>>>>option 1 is better than old')
    elif is_option_2_valid:
        print('only 2nd optoin. comapre to old')
        total_km_old, total_time_old = get_old_routes_total_values(A, route_1, route_2, data)
        if total_km_opt2 < total_km_old and total_time_opt2 <= total_time_old:
            print('>>>>>option 2 is better than old')
    else:
        print('nothing is valid')
        return False

#     to return a bool if swap, if yes i return bool, new route A, route nr A, new route B, new route nr B,

def two_opt_swap(output_df, data, n_iterations):
    A = make_distance_matrix(data)
    routes = df['City Nr.'].tolist()
    routes_of_routes = make_list_containing_lists_routes(routes)

    for i in range(n_iterations):
        nodeA = random.randint(1, 133)
        nodeB = routes[routes.index(nodeA) + 1]
        nodeC = random.randint(1, 133)
        nodeD = routes[routes.index(nodeC) + 1]

        # makes sure that there is no invalid index or edges to swap are the same
        if nodeB > 133 or nodeD > 133 or nodeA == nodeC:
            print('continued')
            continue

        # Basically if the km are lower for routes in total the swap can be made
        # but also the whole time needs to be still in the 8 visit and 10 john work time
        # print(routes_of_routes)
        if are_edges_in_same_route(routes_of_routes, nodeA, nodeC):
            print('in same route')
            print('pass')
            pass

        else:
            check_swap(A, nodeA, nodeB, nodeC, nodeD, output_df, routes_of_routes, data)
            # print('not in same route')


#             here i add everything to output df
#               in output df need to calculate total km again

n_iterations = 100
two_opt_swap(df, data, n_iterations)
