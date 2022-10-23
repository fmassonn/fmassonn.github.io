"""
Author           François Massonnet
Date creation    19 Oct 2022

Script that analyses the long-term statistics of cold situations
"""


# Imports
# -------
import numpy             as np
import matplotlib.pyplot as plt
import os, sys
import matplotlib.dates  as mdates
import matplotlib
matplotlib.rcParams['font.family'] = "Arial Narrow"

import cdsapi
import csv
c = cdsapi.Client()

from datetime import datetime, timedelta
from netCDF4  import Dataset


fileIn = "../output/dailyStatistics_T2m_Bruxelles.csv"

myList = list()
with open(fileIn) as fileCSV:
	csvReader = csv.reader(fileCSV)

	next(csvReader)
	next(csvReader) # Ignore two first lines
	for row in csvReader:
		myList.append([datetime.strptime(row[0], "%Y-%m-%d"), float(row[3])])



# Start analyses
yearb, yeare = 1959, 2021 # Years defining the start of the winter

stats = [[y, len( [ i for i in myList if i[0] >= datetime(y, 7, 1) and i[0] <= datetime(y + 1, 6, 30) and i[1] < 0.0 ] )] for y in np.arange(1959, 2022) ]

# Trend
x = np.array([s[0] for s in stats])
y = np.array([s[1] for s in stats])
p = np.polyfit(x, y, 1)
fit = np.polyval(p, x)

fig, ax  = plt.subplots(1, 1, figsize = (6, 3))
ax.bar([s[0] for s in stats], [s[1] for s in stats])
ax.plot(x, fit, "r--")
ax.grid()
ax.set_title("Nombre de jours avec T$_{max}$ < 0°C\nTendance: " + str(np.round(p[0] * 10, 1)) + " jours / décennie")
ax.set_axisbelow(True)
fig.tight_layout()
plt.savefig("../figs/coldDays.png", dpi = 300)

