"""
Author           François Massonnet
Date creation    3 Oct 2022
Date update      6 Nov 2024, following update in ERA5 API

Script that can be run on a daily basis to download, process
and analyze the 2-m air temperature in some location
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
c = cdsapi.Client()

import requests
import csv
import io

from datetime import datetime, timedelta
from netCDF4  import Dataset


# Functions
# ---------

# Downloading ERA5 data 2-m temp
def downloadERA5(year, listMonths, listDays, listTime, domainArea, outFile = "../data/download.nc"):
    """
	Function to download ERA5 data. Takes as arguments:
	year:       an int, representing a year
	listMonths: a list of months expressed as two-digit strings (e.g. ["04", "12])
	listDays:   same for days
	listTime:   same for time of the day ("03:00")
	domainArea: list of four int representing domain boundaries following ECMWF conventions
	outFile:    the name of the file when it is downloaded
	
	The format is by default to netcdf.
	"""
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
        'product_type': 'reanalysis',
        'format'      : 'netcdf',
        'variable'    : '2m_temperature',
        'year'        : str(int(year)),
        'month'       : listMonths,
        'day'         : listDays,
        'time'        : listTime,
        'area'        : domainArea,
                    },
                    outFile)

# Data with location inforlation
dictLocations = { \
		# Location name Lat max     Lon min  Lat min  Lon max
		"Bruxelles":	 	  [50.8,      4.2,     50.7,    4.3],
		"Sigy-le-Châtel"    :     [46.6,      4.5,     46.5,    4.6],
		}
                    
# ==========================                    
# Editable script parameters


# ERA5 First date of data availability. Should not change except if the 
# data is extended back in time, or if the user does not need the data
# so far back in time
startDate    = datetime(1940, 1, 1)
startYear    = startDate.year

# Domain definition. Fetched from the above dictionary
locationNames = ["Bruxelles", "Sigy-le-Châtel", ]

doAddLatestData = True # Whether to include the RMI latest data (only for Brussels)

# The number of days between today and the latest available data from ERA5
# This number is to be known to identify the time span of the data
lagERA5 = 6

# The years defining the climatology (period of reference)
yearbc, yearec = 1991, 2020

# Kelvin to °C conversion
offsetKtoC = -273.16

# End editable script parameters
# ==============================



for locationName in locationNames:
	try:
        	domainArea = dictLocations[locationName]
	except KeyError:
        	print(locationName + ": Localisation pas encore identifiée")
        	sys.exit()

	# Fetch information regarding today
	today        = datetime.today()
	currentYear  = today.year
	currentDay   = today.day
	currentMonth = today.month
	
	# Define last day of data availability (imposed by ERA5)
	endDate      = today + timedelta(days = - lagERA5)
	endYear      = endDate.year
	endMonth     = endDate.month
	endDay       = endDate.day
	
	
	# Organize input files. There is one file per year
	# Special attention must be paid to the end year
	# because files need to be downloaded separately for the last month
	# (otherwise, there is a crash) and for all months until the last month not included
	
	# The reason it's done this way is (1) to not redownload everything
	# and (2) because NetCDF format differs depending on how lumped the
	# data is. I found that downloading month by month the last year
	# does not cause those issues.
	
	
	# Two list variables that will host the dates and the matching data
	dates = list()
	data  = list()
	
	
	# Run all the years up to the last but one (Python convention)
	for year in [1953]:#np.arange(startYear, endYear):
		fileYear = "../data/download_T2M_" + str(locationName) + "_" + str(year) + ".nc"

		# Check if annual file exists, otherwise run the download function
		if os.path.exists(fileYear):
			print("File " + fileYear + " exists, no download")
		else:
			print("Downloading")
			listMonths = [str(m).zfill(2) for m in np.arange(1, 12 + 1)]
			listDays   = [str(d).zfill(2) for d in np.arange(1, 31 + 1)]
			listTime   = [str(j).zfill(2) + ":00" for j in np.arange(24)]
			downloadERA5(year, listMonths, listDays, listTime, domainArea, outFile = fileYear)
