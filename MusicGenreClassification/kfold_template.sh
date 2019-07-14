#! /bin/sh


logname="placeholder.log"
rm -f $logname
base_command='python kfold.py'
for i in `seq 1 $max`
do
  run_command="$base_command --job_name_suffix _kfold_$i --random_seed $i"
  res=$($run_command)
  run_stat=$(echo "Fold " $i $res)
  echo $run_stat
  echo $run_stat >> $logname
done
