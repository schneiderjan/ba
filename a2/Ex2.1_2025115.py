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
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
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


def are_all_locations_visited(mask):
    mask_list = mask.tolist()
    are_all_visited = all(x == False for x in mask_list)
    return are_all_visited

def get_visiting_time(df, index):
    store_type = df.loc[index].Type
    if store_type == "Jumbo":
        return 1.5
    else:
        return 1


def is_visit_in_time_limit(remaining_travel_time, remaining_visiting_time, time_to_travel, visiting_time, time_to_hq):
    """
    Here we check if the visit can be made time-wise. Three conditions are considered.
    1. is the travel and visit is within johns working limit
    2. is the visit with in the opening hours of the store
    3. can john still make the travel back to the HQ after his visit.
    :param remaining_travel_time:
    :param remaining_visiting_time:
    :param time_to_travel:
    :param visiting_time:
    :param time_to_hq:
    :return: If all conditions are ok true, otherwise, false.
    """
    if (remaining_travel_time - time_to_travel - visiting_time) >= 0 and (
            remaining_visiting_time - time_to_travel - visiting_time) and (
            remaining_travel_time - time_to_hq - time_to_travel - visiting_time) >= 0:
        return True
    else:
        return False


def can_next_location_be_visited(current_index, next_index, remaining_travel_time, remaining_visiting_time, df):
    """
    Determines if the given nr/index of a location can be visited considering the contraints
    on travel time, visiting time, max working hours, etc.
    :param next_index: the next location with minimum travel cost
    :return: True, if location can be visited. False, otherwise.
    """
    distance_to_travel = haversine(df.loc[current_index].Long, df.loc[current_index].Lat, df.loc[next_index].Long,
                                   df.loc[next_index].Lat)
    distance_next_to_hq = haversine(df.loc[next_index].Long, df.loc[next_index].Lat, df.loc[0].Long, df.loc[0].Lat)
    time_to_hq = calculate_travel_time(distance_next_to_hq)
    time_to_travel = calculate_travel_time(distance_to_travel)
    visiting_time = get_visiting_time(df, next_index)

    # The case for the very first visit of the day
    # John considers his travel to arrive exactly at 9h at a given store.
    if remaining_travel_time == 10:
        remaining_travel_time -= time_to_travel
        remaining_travel_time -= visiting_time
        remaining_visiting_time -= visiting_time
        return True, remaining_travel_time, remaining_travel_time, distance_to_travel
    elif remaining_travel_time < 10 and is_visit_in_time_limit(remaining_travel_time, remaining_travel_time,
                                                               time_to_travel, visiting_time, time_to_hq):
        remaining_travel_time -= time_to_travel
        remaining_travel_time -= visiting_time
        remaining_visiting_time -= time_to_travel
        remaining_visiting_time -= visiting_time
        return True, remaining_travel_time, remaining_travel_time, distance_to_travel
    else:
        return False, remaining_travel_time, remaining_travel_time, distance_to_travel


def nearest_neighbor(data, start=0):
    """Nearest neighbor algorithm.
    A is an NxN array indicating distance between N locations
    start is the index of the starting location
    Returns the path and cost of the found solution
    """
    break_flag = False
    A = make_distance_matrix(data)

    remaining_travel_time = 10
    remaining_visiting_time = 8

    route_counter = 0
    total_distance = 0

    N = A.shape[0]
    mask = np.ones(N, dtype=bool)  # boolean values indicating which locations have not been visited
    mask[start] = False

    routes = []

    output_df = pd.DataFrame(
        columns=['Route Nr.', 'City Nr.', 'City Name', 'Total Distance in Route (km)', 'Total distance (km)'])

    while True:
        route_distance = 0
        route = [start]

        output_df = output_df.append(
            {'Route Nr.': route_counter, 'City Nr.': 0, 'City Name': data.loc[0].Name,
             'Total Distance in Route (km)': route_distance, 'Total distance (km)': total_distance},
            ignore_index=True)

        for i in range(N - 1):
            current_index = route[-1]

            # print(mask)
            # print(route)
            if are_all_locations_visited(mask):
                break_flag = True
                end_loc = 0
                route.append(end_loc)
                route_distance += A[current_index, end_loc]
                total_distance += A[current_index, end_loc]
                output_df = output_df.append(
                    {'Route Nr.': route_counter, 'City Nr.': end_loc, 'City Name': data.loc[end_loc].Name,
                     'Total Distance in Route (km)': route_distance, 'Total distance (km)': total_distance},
                    ignore_index=True)
                break

            next_index = np.argmin(A[current_index][mask])  # find minimum of remaining locations

            is_visitable, remaining_travel_time, remaining_travel_time, distance_to_travel = \
                can_next_location_be_visited(current_index, next_index, remaining_travel_time, remaining_visiting_time,
                                             data)
            if is_visitable:
                next_loc = np.arange(N)[mask][next_index]  # convert to original location

                route.append(next_loc)
                mask[next_loc] = False
                route_distance += A[current_index, next_loc]
                total_distance += A[current_index, next_loc]

                output_df = output_df.append(
                    {'Route Nr.': route_counter, 'City Nr.': next_loc, 'City Name': data.loc[next_loc].Name,
                     'Total Distance in Route (km)': route_distance, 'Total distance (km)': total_distance},
                    ignore_index=True)
            elif i == N - 2:
                end_loc = 0
                route.append(end_loc)
                route_distance += A[current_index, end_loc]
                total_distance += A[current_index, end_loc]
                output_df = output_df.append(
                    {'Route Nr.': route_counter, 'City Nr.': end_loc, 'City Name': data.loc[end_loc].Name,
                     'Total Distance in Route (km)': route_distance, 'Total distance (km)': total_distance},
                    ignore_index=True)
            else:
                continue

        print(route)
        routes.append(route)
        route_counter += 1
        remaining_travel_time = 10
        remaining_visiting_time = 8

        if break_flag:
            break

    return output_df


# ex1 nearest neighbour
# EMTE likes to know how many days and how many kilometers John needs in total to visit all the stores.
df = pd.read_excel("Data Excercise 2 - EMTE stores - BA 2019.xlsx")
df['visited'] = False
df['cannot_visit'] = False
df['route_dist'] = 0
df['total_dist'] = 0

print(df.loc[0].Nr)
print(df.loc[0].Name)

route_df = nearest_neighbor(df)

route_df.to_excel('Ex2.1-2025115.xls', index=False)
# when route starts john can travel less than an hour and wait for 9h to start visit.
