from math import radians, cos, sin, asin, sqrt
import pandas as pd

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

# ex1 nearest neighbour
# EMTE likes to know how many days and how many kilometers John needs in total to visit all the stores.
df = pd.read_excel("Data Excercise 2 - EMTE stores - BA 2019.xlsx")

max_working_hours = 10
visiting_time_jumbo = 1.5
visiting_time_others = 1
# visiting time is 9-17

