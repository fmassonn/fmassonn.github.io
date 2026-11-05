#!/bin/bash
source /etc/profile

module purge
source ~/module_load.txt
module load matplotlib/3.10.5-gfbf-2025b

python3 era5_temperature.py

git add -A
git commit -m "Daily update"
git push origin main
