#!/bin/bash

dt=`date '+%m-%d-%H:%M'`
log="training_${dt}.log"
echo $log

CORES=5
rm input_trainsize.txt
touch input_trainsize.txt
for numtrain in `seq 7 13`;
do
    mkdir models${numtrain}
    for trial in `seq 1 10`; 
    do
        mkdir models${numtrain}/trial${trial}
        for model in logreg mnb loglin; 
        do 
            printf "${numtrain}\t${model}\t${trial}\n" >> input_trainsize.txt
        done 
    done
done 

#cat input_trainsize.txt | parallel -v --dryrun --colsep '\t' "python train_master.py {1} {2} {3} &>> ${log}"
cat input_trainsize.txt | parallel --eta -j$CORES --colsep '\t' "python train_master.py {1} {2} {3} &>> ${log}"

