"""
Author           François Massonnet
Date creation    19 Oct 2022

Script that analyses the long-term statistics of min daily temps
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
		print(row)
		myList.append([datetime.strptime(row[0], "%Y-%m-%d"), float(row[2])])



# Min Temps 27 oct
myData = [[m[0].year, m[1]] for m in myList if m[0].month == 10 and m[0].day == 27]

fig, ax  = plt.subplots(1, 1, figsize = (6, 3))

ax.grid()
ax.bar([m[0] for m in myData], [m[1] for m in myData])
ax.set_title("Températures minimales les 27 octobre à Bruxelles (ERA5)")
ax.set_ylabel("$^\circ$ C")
ax.set_ylim(-5.0, 20.0)
ax.set_axisbelow(True)
fig.tight_layout()
plt.savefig("../figs/Tmin_27oct.png", dpi = 300)

