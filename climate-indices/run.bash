#!/bin/bash

module purge
source ~/module_load.txt

python3 era5_temperature.py

git add -A
git commit -m "Daily update"
git push origin main
