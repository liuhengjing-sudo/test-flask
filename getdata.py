# Import Meteostat library and dependencies
from datetime import datetime as dt
import matplotlib.pyplot as plt
from meteostat import Point, Daily, Hourly, Stations
import csv

# Set time period
start = dt(2018, 1, 1)
end = dt(2018, 3, 1)

# Create Point for Vancouver, BC
vancouver = Point(49.2497, -123.1193, 70)

# Get daily data for 2018
data = Hourly(vancouver, start, end)
data = data.fetch()
# print(data)
# print(data[['wdir','wspd']])

# Convert to csv file
# data.to_csv('td.csv',encoding='utf-8')


# Plot line chart including average, minimum and maximum temperature
# data.plot(y=['tavg', 'tmin', 'tmax'])
# data.plot(y=['wdir', 'wspd'])
# plt.show()


def get_wind():
    return data

def get_wind_loc(start_date,end_date,coord):
    coord = coord.replace(" ","").split(',')#Split coord in string format to array
    #Convert date from string format to date format
    start = dt.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    end = dt.strptime(end_date, '%Y-%m-%d %H:%M:%S')

    location = Point(float(coord[0]),float(coord[1]),float(coord[2]))

    data = Hourly(location, start, end)
    data = data.fetch()

    return data





# f = open('testd.csv','w')
# writer = csv.writer(f)
# for i in data:

    # writer.writerow(i)
# f.close()


# Get all stations
# stations = Stations()

# # Get number of stations in northern hemisphere
# northern = stations.bounds((90, -180), (0, 180))
# print('Stations in northern hemisphere:', northern.count())

# # Get number of stations in southern hemisphere
# southern = stations.bounds((0, -180), (-90, 180))
# print('Stations in southern hemisphere:', southern.count())

# # Get random stations in US region
# stations = stations.region('US')
# stations = stations.fetch(10, sample=True)
# print(stations)
# print(stations.region('US'))