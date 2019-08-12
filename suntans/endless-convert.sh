#!/bin/bash

while /bin/true ; do
  date
  python ./ptm_convert.py /shared2/src/sfb_ocean/suntans/runs/merged_018_*  
  sleep 600
done

