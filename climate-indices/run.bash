#!/bin/bash

/opt/easybuild/soft/2025b/software/Python/3.13.5-GCCcore-14.3.0/bin/python3 era5_temperature.py

git add -A
git commit -m "Daily update"
git push origin main
