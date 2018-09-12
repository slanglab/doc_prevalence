#!/bin/bash
CORES=20
numtrain=2000

dt=`date '+%m-%d-%H:%M'`
log="logs/train_prop_${dt}.log"
echo ${log}

rm input_prop.txt
touch input_prop.txt
for prop in `seq 2 10`;
do 
    mkdir models${numtrain}_prop${prop}
    for trial in `seq 1 10`;
    do 
        mkdir models${numtrain}_prop${prop}/trial${trial}

        for model in logreg mnb loglin;
        do
            printf "${numtrain}\t${model}\t${trial}\t${prop}\n" >> input_prop.txt
        done 
    done    
done

cat input_prop.txt | parallel -v --dryrun --colsep '\t' "python train_master.py {1} {2} {3} --prop {4} &>> ${log}"      
cat input_prop.txt | parallel --eta -j$CORES --colsep '\t' "python train_master.py {1} {2} {3} --prop {4} &>> ${log}"      

