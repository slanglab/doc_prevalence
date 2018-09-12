#!/bin/zsh
mkdir results 
mkdir results_prop1
CORES=25 
dt=`date '+%m-%d-%H:%M'`
log="results_${dt}.log"
echo $log

#make input
touch input_master.txt
for model in cc acc pcc mnb loglin hybrid;
do
    for trial in `seq 1 10`;
    do
        printf "${model}\t${trial}\n" >> input_master.txt
    done
done 

#run in parallel
#cat input_master.txt | parallel -v --dryrun --colsep '\t' "python test_with_trials.py {1} {2} &>> ${log}"
cat input_master.txt | parallel --eta -j$CORES --colsep '\t' "python test_with_trials.py {1} {2} &>> ${log}"
cat input_master.txt | parallel --eta -j$CORES --colsep '\t' "python test_with_trials.py {1} {2} --prop &>> ${log}" 



