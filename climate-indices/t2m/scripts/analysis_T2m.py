"""
Author           François Massonnet
Date creation    3 Oct 2022

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
		"Bruxelles":	 [50.8,      4.2,     50.7,    4.3],
		"Cluny"    :     [46.5,      4.6,     46.4,    4.7],
		}
                    
# ==========================                    
# Editable script parameters


# ERA5 First date of data availability. Should not change except if the 
# data is extended back in time, or if the user does not need the data
# so far back in time
startDate    = datetime(1959, 1, 1)
startYear    = startDate.year

# Domain definition. Fetched from the above dictionary
locationName = "Bruxelles"
doAddLatestData = True # Whether to include the RMI latest data (only for Brussels)

try:
	domainArea = dictLocations[locationName]
except KeyError:
	print(locationName + ": Localisation pas encore identifiée")
	sys.exit()

# The origin time in ERA5 (i.e., what their "zero" reference time is.
dateRef = datetime(1900, 1, 1) 

# The number of days between today and the latest available data from ERA5
# This number is to be known to identify the time span of the data
lagERA5 = 6

# The years defining the climatology (period of reference)
yearbc, yearec = 1991, 2020

# Kelvin to °C conversion
offsetKtoC = -273.16

# End editable script parameters
# ==============================





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



# Organize inputfiles. There is one file per year
# Special attention must be paid to end year
# because files need to be downloaded separately for the last month
# (otherwise, crash) and for all months until the last month not included

# The reason it's done this way is (1) to not redownload everything
# and (2) because NetCDF format differs depending on how lumped the
# data is. I found that downloading month by month the last year
# does not cause those issues.


# Two list variables that will host the dates and the matching data
dates = list()
data  = list()


for year in np.arange(startYear, endYear):
	fileYear = "../data/download_T2M_" + str(locationName) + "_" + str(year) + ".nc"
	if os.path.exists(fileYear):
		print("File " + fileYear + " exists, no download")
	else:
		print("Downloading")
		listMonths = [str(m).zfill(2) for m in np.arange(1, 12 + 1)]
		listDays   = [str(d).zfill(2) for d in np.arange(1, 31 + 1)]
		listTime   = [str(j).zfill(2) + ":00" for j in np.arange(24)]
		downloadERA5(year, listMonths, listDays, listTime, domainArea, outFile = fileYear)

	# Read & store the data
	f = Dataset(fileYear, mode = "r")
	thisData = np.squeeze(f.variables["t2m"][:]).data + offsetKtoC # .data to unmask
	thisTime = f.variables["time"][:]
	thisDate = [dateRef + timedelta(days = t / 24) for t in thisTime]
	f.close()


 	# Save the data in the list
	[dates.append(d) for d in thisDate]
	[data.append(d)  for d in thisData]


# Then, treat the special case of the last year
# First, download all months until previous month

# If the month of the last available date is January, then we
# can download all days of that month
if endMonth == 1:
	listMonths = ['01']
	listDays   = [str(d).zfill(2) for d in np.arange(1, endDay + 1)]
	listTime   = [str(j).zfill(2) + ":00" for j in np.arange(24)]
	fileOut    = "../data/download_T2M_" + str(locationName) + "_" + str(endYear) + "-" + str(endMonth).zfill(2) + ".nc"
	downloadERA5(endYear, listMonths, listDays, listTime, domainArea, outFile = fileOut)

	f = Dataset(fileOut, mode = "r")
	thisData = np.squeeze(f.variables["t2m"][:]).data + offsetKtoC # .data to unmask
	thisTime = f.variables["time"][:]
	thisDate = [dateRef + timedelta(days = t / 24) for t in thisTime]
	f.close()


 	# Save the data in array
	[dates.append(d) for d in thisDate]
	[data.append(d)  for d in thisData]
	
else: # if we are in February or later month
	for m in np.arange(1, endMonth):
		listMonths = [str(m).zfill(2)]
		listDays   = [str(d).zfill(2) for d in np.arange(1, 31 + 1)]
		listTime   = [str(j).zfill(2) + ":00" for j in np.arange(24)]
		fileOut = "../data/download_T2M_" + str(locationName) + "_" + str(endYear) + "_" + str(m).zfill(2) + ".nc"
		if os.path.exists(fileOut):
			print("File " + fileOut + " already exists, not downloading")
		else:
			downloadERA5(endYear, listMonths, listDays, listTime, domainArea, outFile = fileOut)

		# Read in the data
		f = Dataset(fileOut, mode = "r")
		thisData = np.squeeze(f.variables["t2m"][:]).data + offsetKtoC # .data to unmask
		thisTime = f.variables["time"][:]
		thisDate = [dateRef + timedelta(days = t / 24) for t in thisTime]
		f.close()
	 	# Save the data in array
		[dates.append(d) for d in thisDate]
		[data.append(d)  for d in thisData]
	
	# Download all days of current month
	listMonths = [str(endMonth).zfill(2)]
	listDays   = [str(d).zfill(2) for d in np.arange(1, endDay + 1)]
	listTime   = [str(j).zfill(2) + ":00" for j in np.arange(24)]
	fileOut = "../data/download_T2M_" + str(locationName) + "_" + str(endYear) + "_" + str(endMonth).zfill(2) + "_" + str(1).zfill(2) + "-" + str(endDay).zfill(2) + ".nc"
	downloadERA5(endYear, listMonths, listDays, listTime, domainArea, outFile = fileOut)

	f = Dataset(fileOut, mode = "r")
	thisData = np.squeeze(f.variables["t2m"][:]).data + offsetKtoC # .data to unmask
	thisTime = f.variables["time"][:]
	thisDate = [dateRef + timedelta(days = t / 24) for t in thisTime]
	f.close()
 	# Save the data in array
	[dates.append(d) for d in thisDate]
	[data.append(d)  for d in thisData]
	



# Check that there is no issue in the date recording: missing day, not evenly spaced data...
if len(set([dates[j + 1] - dates[j] for j, d in enumerate(dates[:-1])])) != 1:
	stop()



# Remove the 29th of Februaries. Those are so annoying and mess up all statistics
data  = [data[j] for j, d in enumerate(dates) if not (d.month == 2 and d.day == 29)]
dates = [d       for    d in dates            if not (d.month == 2 and d.day == 29)]



# Array with years
years = np.arange(startYear, endYear + 1)


# Basic data checks, to make sure nothing is anomalous.
fig, ax = plt.subplots()
ax.plot(dates, data)
fig.savefig("../figs/check.png")
plt.close(fig)

# Write the raw data (hourly) to CSV file
outCSV = "../output/hourly_T2M_" + locationName + ".csv"

with open(outCSV, "w") as csvFile:
	csvFile.write("# Température de l'air à 2 m, fréquence horaire, à " + locationName + " (données ERA5)\n") 
	csvFile.write("AAAA-MM-JJ hh-mm-ss, T2m (°C)\n")
	for j, d in enumerate(dates):
		csvFile.write(str(d) + "," + str(np.round(data[j], 2)) + "\n")	


# Make daily statistics
print("Making daily statistics")
datesDay = dates[::24]

print("... Mean")
dataDayMean =   [np.mean(data[j : j + 24]) for j in np.arange(0, len(data), 24)]
print("... Max")
dataDayMax =   [np.max(data[j : j + 24]) for j in np.arange(0, len(data), 24)]
print("... Min")
dataDayMin =   [np.min(data[j : j + 24]) for j in np.arange(0, len(data), 24)]

# Flag the data as "ERA5"
flagDayData = ["ERA5" for d in dataDayMean]

# Write the data (daily statistics) to CSV file
outCSV = "../output/dailyStatistics_T2m_" + locationName + ".csv"

with open(outCSV, "w") as csvFile:
	csvFile.write("# Statistiques journalières de la température de l'air à 2 m à " + locationName + " (données ERA5)\n") 
	csvFile.write("AAAA-MM-JJ, moyenne (°C), min (°C), max (°C)\n")
	for j, d in enumerate(datesDay):
		csvFile.write(str(d.strftime("%Y-%m-%d")) + "," + str(np.round(dataDayMean[j], 2)) + ","  + \
		                             str(np.round(dataDayMin[j] , 2)) + ","  + \
		                             str(np.round(dataDayMax[j] , 2))        + "\n")


# Special case for Bruxelles
if locationName == "Bruxelles" and doAddLatestData:
    thisURL = "https://www.meteo.be/resources/climatology/uccle_month/Uccle_observations.txt"
    myfile = requests.get(thisURL)
    reader = csv.reader( \
        io.StringIO(myfile.text, newline="\n"), skipinitialspace=True, delimiter=" " \
               )
    # Skip header
    # Skip first rows
    [next(reader) for j in range(5)]
    
    
    datesDayListTemp = list()
    dataDayMeanListTemp = list()
    dataDayMinListTemp = list()
    dataDayMaxListTemp = list()
    
    for j, row in enumerate(reader):
            if j < 10:
                thisDate = datetime.strptime(row[0], "%d-%m-%Y")
                
                if thisDate > datesDay[-1]:
                    datesDayListTemp.append(thisDate)
                    dataDayMeanListTemp.append(row[3])
                    dataDayMinListTemp.append(row[2])
                    dataDayMaxListTemp.append(row[1])
 
    # Append
    [datesDay.append(     d)      for d in datesDayListTemp[   -1::-1]]
    [dataDayMean.append(float(d)) for d in dataDayMeanListTemp[-1::-1 ]]
    [dataDayMin.append(float(d))  for d in dataDayMinListTemp[-1::-1  ]]
    [dataDayMax.append(float(d))  for d in dataDayMaxListTemp[-1::-1  ]]
    [flagDayData.append("RMI")    for d in datesDayListTemp[-1::-1]]



# Data check plots
fig, ax = plt.subplots(2, 2)
ax[0, 0].plot(datesDay, dataDayMean) ; ax[0, 0].set_title("Day mean")
ax[1, 0].plot(datesDay, dataDayMin);    ax[1, 0].set_title("Day min")
ax[1, 1].plot(datesDay, dataDayMax);    ax[1, 1].set_title("Day max")

fig.tight_layout()
fig.savefig("../figs/check2.png")

# Annual outlooks
for year in years:
	print("Annual outlook " + str(year))
	# Subset data
	subDates = [d for d in datesDay if d.year == year]
	subDataMean  = [dataDayMean[j] for j, d in enumerate(datesDay) if d.year == year]
	subDataMin  =  [dataDayMin[j] for j, d in enumerate(datesDay) if d.year == year]
	subDataMax  = [dataDayMax[j] for j, d in enumerate(datesDay) if d.year == year]
	fig, ax = plt.subplots(figsize = (8, 3))
	ax.plot(subDates, subDataMean, linestyle = "-", color = "black", label = "Day mean")
	ax.plot(subDates, subDataMin, linestyle = "-", linewidth = 1, color = "blue", label = "Day min.")
	ax.plot(subDates, subDataMax, linestyle = "-", linewidth = 1, color = "red", label = "Day max.")
	ax.grid()
	ax.set_ylabel("°C")
	ax.set_title("Daily temperature statistics")
	fig.savefig("../figs/outlook_" + str(year) + ".png")
	plt.close(fig)

# Make annual statistics
print("Making annual statistics")
print("... Mean")
dataYearMean = [np.mean([data[jj] for jj, dd in enumerate(dates) if dd.year == y]) for y in years]

# Data check plots
fig, ax = plt.subplots()
ax.plot(years, dataYearMean)
ax.grid()
ax.set_title("Warning: last year is not finished")
fig.savefig("../figs/check3.png")
plt.close(fig)



# Compute seasonal cycle
# Ref date (no leap)
print("... Computing annual cycle")
dateOneYear = [dateRef + j * timedelta(days = 1) for j in np.arange(365)]


cycle = [np.mean([dataDayMean[j] for j, dd in enumerate(datesDay) if \
		dd.day   == d.day    and \
		dd.month == d.month  and \
		dd.year >= yearbc    and \
		dd.year <= yearec        \
		]) for d in dateOneYear]
cycle = np.array(cycle)

# Smoothed cycle
print("... Smoothing & tiling cycle")
widthSmooth = 61
cycleSmoothed = [np.mean([cycle[k] for k in [(365 + d - j) % 365 for j in range(-int(widthSmooth / 2), int(widthSmooth / 2) + 1)]]) for d in np.arange(len(cycle))]
cycleSmoothedTiled = [cycleSmoothed[j % 365] for j in np.arange(len(datesDay))]

print("... Producing figures")
# Data check plots
fig, ax = plt.subplots()
ax.plot(dateOneYear, cycle)
ax.plot(dateOneYear, cycleSmoothed, "k--")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.grid()
ax.set_title("Annual cycle " + str(yearbc) + "-" + str(yearec))
fig.savefig("../figs/check4.png")
plt.close(fig)


# Plot previous 365 days and previous years

fig, ax = plt.subplots(figsize = (7, 4))

# Tile the cycle
# Subset the data
ax.plot(datesDay, cycleSmoothedTiled, lw = 1, color = "k", ls = "--", label = "Moyenne climatologique (" + str(yearbc) + "-" + str(yearec) + ")")

#ax.plot(dateOneYear, cycle_smoothed, "k--")
anomalies = np.array(dataDayMean) - np.array(cycleSmoothedTiled)
for j, d in enumerate(datesDay):
	if d > today - timedelta(days = 365):
		xmin, xmax = -10, 10
		color = plt.cm.RdBu_r(int((anomalies[j]- xmin) * 255 / (xmax - xmin)))[:3]
		if flagDayData[j] == "ERA5":
			ax.bar(datesDay[j], anomalies[j], bottom = cycleSmoothedTiled[j], color = color, lw = 2, width = 1.0)
		else:
			ax.bar(datesDay[j], anomalies[j], bottom = cycleSmoothedTiled[j], color = "white", edgecolor = color, linestyle = "--", lw = 0.25, width = 1.0)
            
            
locator = mdates.MonthLocator()  # every month
ax.grid()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %y'))
ax.xaxis.set_tick_params(rotation=45)
ax.set_axisbelow(True)
ax.set_xlim(today - timedelta(days = 365), today + timedelta(days = 10))
ax.set_ylim(-10, 35)
ax.plot((-1e9, 1e9), (0, 0), color = "black")
ax.set_ylabel("$^\circ$ C")
ax.set_title("Température journalière moyenne de l'air à 2 m\n" + locationName + " (données ERA5)")
ax.text(ax.get_xlim()[1], ax.get_ylim()[0],  "\nDernière donnée: " + str(datesDay[-1].strftime("%d %b %y"))  +
            ". Graphe: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
            " | @FMassonnet",
            rotation =90, ha = "left", va = "bottom", fontsize = 4)
ax.legend()
fig.tight_layout()
fig.savefig("../figs/T2m_" + locationName + "_" + "last365d.png", dpi = 300)
plt.close(fig)


# Repeat for all years before this year
for year in np.arange(startYear, endYear):
	print(year)
	fig, ax = plt.subplots(figsize = (7, 4))

	# Subset the data
	ax.plot(datesDay, cycleSmoothedTiled, lw = 1, color = "k", ls = "--", label = "Moyenne climatologique (" + str(yearbc) + "-" + str(yearec) + ")")

	#ax.plot(dateOneYear, cycle_smoothed, "k--")
	anomalies = np.array(dataDayMean) - np.array(cycleSmoothedTiled)
	for j, d in enumerate(datesDay):
		if d >= datetime(year, 1, 1) and d <= datetime(year, 12, 31):
			xmin, xmax = -10, 10
			color = plt.cm.RdBu_r(int((anomalies[j]- xmin) * 255 / (xmax - xmin)))[:3]
			ax.bar(datesDay[j], anomalies[j], bottom = cycleSmoothedTiled[j], color = color, lw = 2, width = 1.0)
	locator = mdates.MonthLocator()  # every month
	ax.grid()
	ax.xaxis.set_major_locator(locator)
	ax.set_xlim(datetime(year - 1, 12, 31), datetime(year + 1, 1, 1))
	ax.set_ylim(-10, 35)
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %y'))
	ax.xaxis.set_tick_params(rotation=45)
	ax.set_axisbelow(True)
	ax.plot((-1e9, 1e9), (0, 0), color = "black")
	ax.set_ylabel("$^\circ$ C")
	ax.set_title("Température journalière moyenne de l'air à 2 m\n" + "Année " + str(year) + ", " + locationName + " (données ERA5)")
	ax.legend()
	fig.tight_layout()
	fig.savefig("../figs/T2m_" + locationName + "_" + str(year) + ".png", dpi = 300)
	plt.close(fig)



# Do the max-min statistics
# -------------------------

fig, ax = plt.subplots(figsize = (7, 4))

# The maximum recorded for that day until now (without today). The first year we can't do anything obviously.
maxToDate = [np.nan for _ in range(365)] + [np.max(dataDayMax[j % 365 : j : 365]) for j in np.arange(365, len(dataDayMax))]
minToDate = [np.nan for _ in range(365)] + [np.min(dataDayMin[j % 365 : j : 365]) for j in np.arange(365, len(dataDayMin))]

doLegend = True
doLegendRecordMax = True
doLegendRecordMin = True

for j, d in enumerate(datesDay):
        if d > today - timedelta(days = 365):
        
                if doLegend:
                    legendMax="Maximum journalier à ce jour"
                    legendMin="Minimum journalier à ce jour"
                    legendRange="Min-max du jour"

                    
                    doLegend = False
                else:
                    legendMax = legendMin = legendRange = legendRange = legendRecordMax = legendRecordMin = None
                    
                    
                xmin, xmax = 0, 50
                color = plt.cm.OrRd(int((maxToDate[j]- xmin) * 255 / (xmax - xmin)))[:3]
                ax.bar(datesDay[j], maxToDate[j] - cycleSmoothedTiled[j], bottom = cycleSmoothedTiled[j], color = color, lw = 2, width = 1, alpha = 0.9, label = legendMax)
                
                xmin, xmax = -10, 20
                color = plt.cm.GnBu_r(int((minToDate[j]- xmin) * 255 / (xmax - xmin)))[:3]
                ax.bar(datesDay[j], cycleSmoothedTiled[j] - minToDate[j], bottom = minToDate[j] , color = color, lw = 2, width = 1, alpha = 0.9, label = legendMin)
                
                ax.bar(datesDay[j], dataDayMax[j] - dataDayMin[j], bottom = dataDayMin[j] , color = [0.2, 0.2, 0.2], lw = 1, width = 0.8, alpha = 1, label = legendRange)
                
                if dataDayMax[j] > maxToDate[j]:
                    if doLegendRecordMax:
                        legendRecordMax = "Record max"
                        doLegendRecordMax = False
                    else:
                        legendRecordMax = None
                    ax.scatter(datesDay[j], dataDayMax[j], 10, marker = "*", color = "r", lw = 0, zorder = 1000, label = legendRecordMax)

                if dataDayMin[j] < minToDate[j]:
                    if doLegendRecordMin:
                        legendRecordMin = "Record min"
                        doLegendRecordMin = False
                    else:
                        legendRecordMin = None
                    ax.scatter(datesDay[j], dataDayMin[j], 10, marker = "*", color = "b", lw = 0, zorder = 1000, label = legendRecordMin)
                    

                    
                
#ax.plot(datesDay, dataDayMax, lw = 0.5, color = "k")
#ax.plot(datesDay, dataDayMin, lw = 0.5, color = "k")

ax.plot(datesDay, cycleSmoothedTiled, lw = 0.5, color = "k", ls = "-", label = "Moyenne climatologique journalière (" + str(yearbc) + "-" + str(yearec) + ")")
locator = mdates.MonthLocator()  # every month
ax.grid()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %y'))
ax.xaxis.set_tick_params(rotation=45)
ax.set_axisbelow(True)
ax.set_xlim(today - timedelta(days = 365), today + timedelta(days = 10))
ax.set_ylim(-25, 40)
ax.plot((-1e9, 1e9), (0, 0), color = "black")
ax.set_ylabel("$^\circ$ C")
ax.set_title("Températures minimales et maximales journalières de l'air à 2 m\n" + locationName + " (données ERA5)")
ax.text(ax.get_xlim()[1], ax.get_ylim()[0],  "\nDernière donnée: " + str(datesDay[-1].strftime("%d %b %y"))  +
            ". Graphe: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
            " | @FMassonnet",
            rotation =90, ha = "left", va = "bottom", fontsize = 4)
ax.legend(fontsize = 6)
fig.tight_layout()
fig.savefig("../figs/T2m_MinMax_" + locationName + "_" + "last365d.png", dpi = 300)
plt.close(fig)



# Repeat for all years before this year
# starting only at year startYear + 1 since no max/min for first year
#for year in np.arange(startYear + 1, endYear):
#	print("year")
#	fig, ax = plt.subplots(figsize = (7, 4))
#    
#	doLegend = True
#	doLegendRecordMax = True
#	doLegendRecordMin = True
#
#	for j, d in enumerate(datesDay):
#		if d >= datetime(year, 1, 1) and d <= datetime(year, 12, 31):
#			if doLegend:
#			    legendMax="Maximum journalier à ce jour"
#			    legendMin="Minimum journalier à ce jour"
#			    legendRange="Min-max du jour"
#
#			    
#			    doLegend = False
#			else:
#			    legendMax = legendMin = legendRange = legendRange = legendRecordMax = legendRecordMin = None
#			    
#			    
#			xmin, xmax = 0, 50
#			color = plt.cm.OrRd(int((maxToDate[j]- xmin) * 255 / (xmax - xmin)))[:3]
#			ax.bar(datesDay[j], maxToDate[j] - cycleSmoothedTiled[j], bottom = cycleSmoothedTiled[j], color = color, lw = 2, width = 1, alpha = 0.9, label = legendMax)
#			
#			xmin, xmax = -10, 20
#			color = plt.cm.GnBu_r(int((minToDate[j]- xmin) * 255 / (xmax - xmin)))[:3]
#			ax.bar(datesDay[j], cycleSmoothedTiled[j] - minToDate[j], bottom = minToDate[j] , color = color, lw = 2, width = 1, alpha = 0.9, label = legendMin)
#			
#			ax.bar(datesDay[j], dataDayMax[j] - dataDayMin[j], bottom = dataDayMin[j] , color = [0.2, 0.2, 0.2], lw = 1, width = 0.8, alpha = 1, label = legendRange)
#			
#			if dataDayMax[j] > maxToDate[j]:
#			    if doLegendRecordMax:
#				legendRecordMax = "Record max"
#				doLegendRecordMax = False
#			    else:
#				legendRecordMax = None
#			    ax.scatter(datesDay[j], dataDayMax[j], 10, marker = "*", color = "r", lw = 0, zorder = 1000, label = legendRecordMax)
#
#			if dataDayMin[j] < minToDate[j]:
#			    if doLegendRecordMin:
#				legendRecordMin = "Record min"
#				doLegendRecordMin = False
#			    else:
#				legendRecordMin = None
#			    ax.scatter(datesDay[j], dataDayMin[j], 10, marker = "*", color = "b", lw = 0, zorder = 1000, label = legendRecordMin)
#			    
#
#			    
#			
#	#ax.plot(datesDay, dataDayMax, lw = 0.5, color = "k")
#	#ax.plot(datesDay, dataDayMin, lw = 0.5, color = "k")
#
#	ax.plot(datesDay, cycleSmoothedTiled, lw = 0.5, color = "k", ls = "-", label = "Moyenne climatologique journalière (" + str(yearbc) + "-" + str(yearec) + ")")
#	locator = mdates.MonthLocator()  # every month
#	ax.grid()
#	ax.xaxis.set_major_locator(locator)
#	ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %y'))
#	ax.xaxis.set_tick_params(rotation=45)
#	ax.set_axisbelow(True)
#	ax.set_xlim(datetime(year - 1, 12, 31), datetime(year + 1, 1, 1))
#	ax.set_ylim(-25, 40)
#	ax.plot((-1e9, 1e9), (0, 0), color = "black")
#	ax.set_ylabel("$^\circ$ C")
#	ax.set_title("Températures minimales et maximales journalières de l'air à 2 m\n" +  "Année " + str(year) + ", "locationName + " (données ERA5)")
#	ax.text(ax.get_xlim()[1], ax.get_ylim()[0],  "\nDernière donnée: " + str(datesDay[-1].strftime("%d %b %y"))  +
#		    ". Graphe: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
#		    " | @FMassonnet",
#		    rotation =90, ha = "left", va = "bottom", fontsize = 4)
#	ax.legend(fontsize = 8)
#	fig.tight_layout()
#	fig.savefig("../figs/T2m_MinMax" + locationName + "_" + str(year) + ".png", dpi = 300)
#	plt.close(fig)
#
#
#
