import pandas as pd
from math import radians, cos, sin, asin, sqrt
import numpy as np
import random


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
    """
    Creates a list that contains all routes as a list
    :param routes:
    :return:
    """
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


def are_edges_in_same_route(routes_of_routes, node_a, node_c):
    for idx, route in enumerate(routes_of_routes):
        if node_a in route and node_c in route:
            return True
    return False


def recalculate_route(A, route, data):
    """
    Recalculates a given route to verify constraints
    :param A: distance matrix
    :param route: the given route
    :param data: data from assignment
    :return:
    """
    total_travel_time = 0
    total_visiting_time = 0
    total_km = 0
    kms = [0]

    current_loc = route[0]
    n = len(route)
    for idx in range(1, n):
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
        elif idx == n - 1:
            next_loc = route[idx]

            travel_dist = A[current_loc, next_loc]
            travel_time = calculate_travel_time(travel_dist)
            total_travel_time += travel_time
            total_visiting_time += travel_time

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
            return False, -1, []
    return True, total_km, kms


def get_old_routes_total_values(A, route_1, route_2, data):
    if len(route_2) > 0:
        is_valid, total_km_1, kms_1 = recalculate_route(A, route_1, data)
        is_valid, total_km_2, kms_2 = recalculate_route(A, route_2, data)

        return total_km_1 + total_km_2
    else:
        is_valid, total_km, kms = recalculate_route(A, route_1, data)
        return total_km


def check_swap_two_routes(A, nodeA, nodeC, routes_of_routes, data):
    """
    If two nodes from different routes are selected then we build both options of swapping
    and then check which one is better and perform exchange if possible
    :param A:
    :param nodeA:
    :param nodeC:
    :param routes_of_routes:
    :param data:
    :return:
    """
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

    # option1
    total_km_opt1 = -1
    is_option_1_valid = False
    is_route_valid_ad, total_km_ad, kms_ad = recalculate_route(A, new_route_ad, data)
    is_route_valid_bc, total_km_bc, kms_bc = recalculate_route(A, new_route_bc, data)
    if is_route_valid_ad and is_route_valid_bc:
        is_option_1_valid = True
        total_km_opt1 = total_km_ad + total_km_bc

    # option2
    total_km_opt2 = -1
    is_option_2_valid = False
    is_route_valid_ac, total_km_ac, kms_ac = recalculate_route(A, new_route_ac, data)
    is_route_valid_bd, total_km_bd, kms_bd = recalculate_route(A, new_route_bd, data)
    if is_route_valid_ac and is_route_valid_bd:
        is_option_2_valid = True
        total_km_opt2 = total_km_ac + total_km_bd

    # check if an option is better than the original route.
    total_km_old = get_old_routes_total_values(A, route_1, route_2, data)
    if is_option_1_valid and total_km_opt1 < total_km_old:
        if not is_option_2_valid:
            return True, new_route_ad, new_route_bc, route_nr_a, route_nr_c

        elif is_option_2_valid and total_km_opt1 < total_km_opt2:
            return True, new_route_ad, new_route_bc, route_nr_a, route_nr_c

    elif is_option_2_valid and total_km_opt2 < total_km_old:
        return True, new_route_ac, new_route_bd, route_nr_a, route_nr_c

    else:
        return False, [], [], -1, -1


def check_swap_one_route(A, node_a, node_c, routes_of_routes, data):
    """
    If two nodes are selected in the same route this method verifies whether a swap can be made
    :param A: distance matrix
    :param node_a: city 1
    :param node_c: city 2
    :param routes_of_routes: list of all routes
    :param data: data from assignment
    :return:
    """
    route_ = []
    route_nr = -1

    for route in routes_of_routes:
        if node_a in route:
            route_ = route
            route_nr = routes_of_routes.index(route_)
            break

    idx_a = route_.index(node_a)
    idx_c = route_.index(node_c)

    index_diff = abs(idx_a - idx_c)
    new_route = []
    if index_diff >= 3:
        if idx_a > idx_c:
            idx_d = idx_a - 1
            new_route = route_[:idx_c + 1] + route_[idx_d:idx_c:-1] + route_[idx_a:]
        else:
            idx_b = idx_c - 1
            new_route = route_[:idx_a + 1] + route_[idx_b: idx_a:-1] + route_[idx_c:]
    elif index_diff == 1:
        new_route = route_
        new_route[idx_a] = node_c
        new_route[idx_c] = node_a
    elif index_diff == 2:
        if idx_a > idx_c:
            new_route = route_[:idx_c] + route_[idx_a:idx_c - 1:-1] + route_[idx_a + 1:]
        else:
            new_route = route_[:idx_a] + route_[idx_c:idx_a - 1:-1] + route_[idx_c + 1:]

    if len(new_route) > 0:
        is_route_valid, total_km, kms = recalculate_route(A, new_route, data)
        total_old_km = get_old_routes_total_values(A, route_, [], data)
        if is_route_valid and total_km < total_old_km:
            return True, new_route, route_nr

    return False, [], -1


def create_data_to_append(new_route_1, route_nr_1, kms_1, new_route_2, route_nr_2, kms_2):
    new_data = []
    for i in range(len(new_route_1)):
        new_data.append([route_nr_1, new_route_1[i], 'city_tbd', kms_1[i], -1])
    if len(new_route_2) > 0:
        for j in range(len(new_route_2)):
            new_data.append([route_nr_2, new_route_2[j], 'city_tbd', kms_2[j], -1])

    return new_data


def create_new_df(A, routes_of_routes, data):
    total_distance = 0
    new_data = []
    print(routes_of_routes)
    for route_idx in range(len(routes_of_routes)):
        route_distance = 0
        previous_loc = 0

        for loc_idx in range(len(routes_of_routes[route_idx])):
            route = routes_of_routes[route_idx]
            loc = route[loc_idx]
            if loc_idx == 0:
                dist = 0
            else:
                dist = A[previous_loc, loc]

            route_distance += dist
            total_distance += dist
            city_name = data.iat[loc, 1]
            previous_loc = loc

            new_data.append([route_idx, loc, city_name, route_distance, total_distance])

        print('Total distance: {}'.format(total_distance))

    return pd.DataFrame(new_data, columns=['Route Nr.', 'City Nr.', 'City Name', 'Total Distance in Route (km)',
                                           'Total distance (km)'])


def two_opt_swap(data, n_iterations):
    A = make_distance_matrix(data)
    # tabu_matrix = make_tabu_matrix(data)
    routes = df['City Nr.'].tolist()
    routes_of_routes = make_list_containing_lists_routes(routes)

    for i in range(n_iterations):
        if i % 1000 == 0:
            print('Iteration {}'.format(i))

        node_a = random.randint(1, 133)
        node_c = random.randint(1, 133)

        # makes sure that there is no invalid index to get an index out of range exception
        # or edges to swap are the same
        if node_a == node_c:
            continue

        # Basically if the km are lower for routes in total the swap can be made
        # but also the whole time needs to be still in the 8 visit and 10 john work time
        if are_edges_in_same_route(routes_of_routes, node_a, node_c):
            is_swap_good, new_route, route_nr = check_swap_one_route(A, node_a, node_c, routes_of_routes, data)
            if is_swap_good:
                routes_of_routes[route_nr] = new_route
        else:
            is_swap_good, new_route_1, new_route_2, route_nr_1, route_nr_2 = check_swap_two_routes(A, node_a, node_c,
                                                                                                   routes_of_routes,
                                                                                                   data)
            if is_swap_good:
                routes_of_routes[route_nr_1] = new_route_1
                routes_of_routes[route_nr_2] = new_route_2

    return create_new_df(A, routes_of_routes, data)


df = pd.read_excel('Ex2.1-2025115.xls')
data = pd.read_excel('Data Excercise 2 - EMTE stores - BA 2019.xlsx')

n_iterations = 50000
output_df = two_opt_swap(data, n_iterations)
output_df.to_excel('Ex2.2-2025115.xls', index=False)
