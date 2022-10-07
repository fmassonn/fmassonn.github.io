#!/bin/bash

python3 analysis_T2m.py


git add ../figs/T2m_Bruxelles_*.png


git commit -m "Daily update"

git push origin main
