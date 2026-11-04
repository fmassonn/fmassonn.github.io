#!/bin/bash


/opt/easybuild/soft/2025b/software/ELIC_Python/1-foss-2025b/bin/python3 era5_temperature.py

git add -A
git commit -m "Daily update"
git push origin main
