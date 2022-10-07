#!/bin/bash

#python3 analysis_T2m.py


#git add ../figs/T2m_Bruxelles_*.png

# Create page with all years
thisYear=`date +"%Y"`

echo "<p align=\"center\">" > ../allYears.md

for year in `seq $(( $thisYear - 1)) 1959`
do
	echo $year
	echo "<h1> $year </h1>" >> ../allYears.md
	echo "<img src=\"./figs/T2m_Bruxelles_$year.png\" width=\"1200\">" >> ../allYears.md
	echo "<br></br>" >> ../allYears.md
done

echo "</p>" >> ../allYears.md


#git commit -m "Daily update"

#git push origin main
