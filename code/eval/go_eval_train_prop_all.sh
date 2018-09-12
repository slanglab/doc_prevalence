#!/bin/bash
CORES=20
numtrain=2000

dt=`date '+%m-%d-%H:%M'`
log="logs/train_prop_${dt}.log"
echo ${log}

rm input_prop.txt
touch input_prop.txt
for prop in `seq 2 9`;
do 
    mkdir results_prop${prop}
    for trial in `seq 1 10`;
    do 
        for model in cc acc pcc mnb loglin hybrid;
        do
            printf "${model}\t${trial}\t${prop}\n" >> input_prop.txt
        done 
    done    
done

#cat input_master.txt | parallel -v --dryrun --colsep '\t' "python test_with_trials.py {1} {2} --prop {3} &>> ${log}"
cat input_prop.txt | parallel --eta -j$CORES --colsep '\t' "python test_with_trials.py {1} {2} --prop {3} &>> ${log}"