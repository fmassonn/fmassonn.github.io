#!/bin/bash

#!/bin/bash

cd /home/elic/fmasson/git/fmassonn.github.io/climate-indices

export PYTHONPATH=/opt/easybuild/soft/2025b/software/ELIC_Python/1-foss-2025b/lib/python3.13/site-packages

python3 era5_temperature.py

git add -A
git commit -m "Daily update"
git push origin main
