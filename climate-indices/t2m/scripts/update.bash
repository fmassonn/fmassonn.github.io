#!/bin/bash

python3 analysis_T2m.py

git add ../figs/last_1yr.png ../figs/last_20yr.png
git commit -m "Daily update"

git push origin main
