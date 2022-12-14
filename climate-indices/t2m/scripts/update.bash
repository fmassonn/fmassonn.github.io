#!/bin/bash -l

set -o nounset
set -o errexit
#set -x
if [[ $HOSTNAME == "aurora" ]]
then
	module purge
	module load releases/2020b
	module load ELIC_Python/1-foss-2020b
fi

#python3 analysis_T2m.py



# Create page with all years
thisYear=`date +"%Y"`

outFile="../T2mAllYears.md"

echo "<p align=\"center\">" > $outFile
for year in `seq $(( $thisYear - 1)) -1 1959`
do
	echo $year
	echo "<h1> $year </h1>" >> $outFile
	echo "<img src=\"./figs/T2m_Bruxelles_$year.png\" width=\"1200\">" >> $outFile
	echo "<br>" >> $outFile
done

echo "</p>" >> $outFile

git add ../figs/T2m_Bruxelles_*.png
git add ../figs/T2m_MinMax_Bruxelles_last365d.png
git add ../output/hourly_T2M_Bruxelles.csv
git add ../output/dailyStatistics_T2m_Bruxelles.csv
git add ../output/hourly_T2M_Bruxelles.csv
git commit -m "Daily update"

git push origin main
