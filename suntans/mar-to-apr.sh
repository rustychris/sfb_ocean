#!/bin/bash

PFX=/opt2/sfb_ocean/suntans/runs/merge_017

date
echo "June"
[ -d ${PFX}-201706 ] || python ./merged_sun.py -s 2017-06-01T00:00:00 -d ${PFX}-201706 -e 2017-07-01T12:00:00 || exit 1
date
echo "July"
[ -d ${PFX}-201707 ] || python ./merged_sun.py -r ${PFX}-201706 -d ${PFX}-201707 -e 2017-08-01T12:00:00 || exit 1
date
echo "August"
[ -d ${PFX}-201708 ] || python ./merged_sun.py -r ${PFX}-201707 -d ${PFX}-201708 -e 2017-09-01T12:00:00 || exit 1
date
echo "September"
[ -d ${PFX}-201709 ] || python ./merged_sun.py -r ${PFX}-201708 -d ${PFX}-201709 -e 2017-10-01T12:00:00 || exit 1
date
echo "October"
[ -d ${PFX}-201710 ] || python ./merged_sun.py -r ${PFX}-201709 -d ${PFX}-201710 -e 2017-11-01T12:00:00 || exit 1
date
echo "November"
[ -d ${PFX}-201711 ] || python ./merged_sun.py -r ${PFX}-201710 -d ${PFX}-201711 -e 2017-12-01T12:00:00 || exit 1
date
echo "December"
[ -d ${PFX}-201712 ] || python ./merged_sun.py -r ${PFX}-201711 -d ${PFX}-201712 -e 2018-01-01T12:00:00 || exit 1 
date
echo "January"
[ -d ${PFX}-201801 ] || python ./merged_sun.py -r ${PFX}-201712 -d ${PFX}-201801 -e 2018-02-01T12:00:00 || exit 1
date
echo "February"
[ -d ${PFX}-201802 ] || python ./merged_sun.py -r ${PFX}-201801 -d ${PFX}-201802 -e 2018-03-01T12:00:00 || exit 1
date
echo "March"
[ -d ${PFX}-201803 ] || python ./merged_sun.py -r ${PFX}-201802 -d ${PFX}-201803 -e 2018-04-01T12:00:00 || exit 1
date
echo "April"
[ -d ${PFX}-201804 ] || python ./merged_sun.py -r ${PFX}-201803 -d ${PFX}-201804 -e 2018-05-01T12:00:00 || exit 1
date
echo "May"
[ -d ${PFX}-201805 ] || python ./merged_sun.py -r ${PFX}-201804 -d ${PFX}-201805 -e 2018-06-01T12:00:00 || exit 1
date
echo "June"
[ -d ${PFX}-201806 ] || python ./merged_sun.py -r ${PFX}-201805 -d ${PFX}-201806 -e 2018-07-01T12:00:00 || exit 1

