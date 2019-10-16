#!/bin/bash
# This script requires an installation of the Chadwick Baseball Bureau software

cd ../data/retrosheet/playbyplay/
year=1919
until [ $year -gt 2018 ]
do
  cd $year
  touch "all$year.csv"
  cwevent -y $year -f 0-96 $year*.EV* > playbyplay$year.csv
  ((year++))
  cd ..
done

echo Chadwick finished