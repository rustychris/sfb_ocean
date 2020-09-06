#!/bin/bash

# This needs to be the component specific jobid, so add 1 maybe.
JOB=$1

NODELIST=`squeue -u rustyh -h -j $JOB -o '%N'`

HOSTS=`scontrol show hostnames $NODELIST`

for HOST in $HOSTS ; do
    echo $HOST
    PIDS=`ssh $HOST ps h -o pid -C sun`

    for PID in $PIDS ; do
	echo ________  PID=$PID HOST=$HOST ________________ 
	ssh $HOST gdb -q $HOME/src/suntans/main/sun <<EOF
attach $PID
where -7
detach
EOF
	echo
    done
done

