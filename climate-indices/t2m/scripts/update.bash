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

python3 analysis_T2m.py


# Create master page
for locName in "Bruxelles" "Sigy-le-Châtel"
do
  sed -e "s/LLOOCCNNAAMMEE/$locName/" ../skeleton.md > ../$locName.md



  # Create page with all years
  thisYear=`date +"%Y"`

  outFile="../T2mAllYears_$locName.md"

  echo "<p align=\"center\">" > $outFile
  for year in `seq $(( $thisYear - 1)) -1 1959`
  do
  	echo $year
  	echo "<h1> $year </h1>" >> $outFile
  	echo "<img src=\"./figs/T2m_${locName}_$year.png\" width=\"1200\">" >> $outFile
  	echo "<br>" >> $outFile
  done

  echo "</p>" >> $outFile

  git add ../figs/T2m_${locName}_*.png
  git add ../figs/T2m_MinMax_${locName}_last365d.png
 
  git add ../T2mAllYears_${locName}.md
  git add ../${locName}.md

  git add ../output/hourly_T2M_${locName}.csv
  git add ../output/dailyStatistics_T2m_${locName}.csv
  git add ../output/hourly_T2M_${locName}.csv

done

git commit -m "Daily update"

git push origin operational
