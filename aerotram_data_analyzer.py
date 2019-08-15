import pyreadr
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from route import Route
from waySet import WaySet


def filter_sector(df, sector):
    f = df["Latitude"] > sector[0]
    f = f & (df["Latitude"] < sector[2])
    f = f & (df["Longitude"] > sector[1])
    f = f & (df["Longitude"] < sector[3])
    return df[f]


def analyze_dataframe(name, df, sector):
    df = filter_sector(df, sector)

    if (len(df.index) > 0):
        drives = np.unique(df["#Drive"].values)
        # print(drives)
        for d in drives:
            mask = df["#Drive"] == d
            drive_name = "%s_Drive_%d" % (name, d)
            drive_data = df[mask]
            analyze_drive(drive_name, drive_data)


def analyze_drive(drive_name, drive_data):

    print("Processing " + drive_name)

    n_sec = 1
    drive_data = drive_data.iloc[::n_sec, :]

    route = Route(latitudes=drive_data["Latitude"], longitudes=drive_data["Longitude"])

    indices = route.not_too_close_points_indices(0.0005)
    drive_data = drive_data.iloc[indices, :]

    route = Route(latitudes=drive_data["Latitude"], longitudes=drive_data["Longitude"])

    route_sector = route.get_sector()
    way_set = WaySet.download_all_ways(route_sector, onlyTram=False)
    snapped_route = route.snap_points(way_set)

    latitudes, longitudes = Route.get_lat_lon_from_points(snapped_route.points)
    drive_data.insert(0, "Snapped_Lat", latitudes)
    drive_data.insert(0, "Snapped_Lon", longitudes)

    drive_data.insert(0, "Heading", snapped_route.headings)  # Check length

    drive_data.to_csv("Results/%s.csv" % drive_name)

    wps = [{"Time": row["Time"],
            "Lat": row["Snapped_Lat"],
            "Lon": row["Snapped_Lon"],
            "Heading": row["Heading"],
            "Value": row["1MinMean"]} for index, row in drive_data.iterrows()]
    with open("Results/%s.json" % drive_name, 'w') as outfile:
        json.dump(wps, outfile, indent=4, sort_keys=True)

    route.plot(style='or')
    snapped_route.plot()
    way_set.plot()
    #plt.show()
    plt.savefig("Results/" + drive_name + ".svg")
    plt.clf()


################################

ka_ost_sector = (48.999195, 8.401027, 49.019584, 8.434186)
folder = "Results"
if not os.path.isdir(folder):
    os.mkdir(folder)
result = pyreadr.read_r('Data/TramData_CalenderWeek_50.RData')  # also works for Rds
print(result.keys())  # let's check what objects we got

for dataframe_name in result:
    analyze_dataframe(dataframe_name, result[dataframe_name], sector=ka_ost_sector)