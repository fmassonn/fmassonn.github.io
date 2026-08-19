#!/bin/bash

for year in `seq 1940 2023`
do
sed -e "s/__YEAR__/${year}/" shortcut.py > sk_${year}.py
nohup /usr/local/bin/python3 sk_${year}.py >& log_$year & 
sleep 1
done
