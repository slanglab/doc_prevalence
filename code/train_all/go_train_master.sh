#!/bin/bash
CORES=20
numtrain=2000

mkdir models${numtrain}
mkdir models${numtrain}_prop1
mkdir logs

dt=`date '+%m-%d-%H:%M'`
log="logs/train_prop_${dt}.log"
echo ${log}

for trial in `seq 1 10`;
    do
        mkdir models${numtrain}/trial${trial}
        mkdir models${numtrain}_prop1/trial${trial}
    done

#make input 
rm input_parallel.txt
touch input_parallel.txt

for model in logreg mnb loglin; 
do
    for trial in `seq 1 10`;
    do
        printf "${numtrain}\t${model}\t${trial}\n" >> input_parallel.txt
    done
done 

#cat input_parallel.txt | parallel -v --dryrun --colsep '\t' "python train_master.py {1} {2} {3} --prop"  
cat input_parallel.txt | parallel --eta -j$CORES --colsep '\t' "python train_master.py {1} {2} {3} &>> ${log}"
cat input_parallel.txt | parallel --eta -j$CORES --colsep '\t' "python train_master.py {1} {2} {3} --prop &>> ${log}"      

