#!/bin/zsh
dt=`date '+%m-%d-%H:%M'`
log="logs/results_trainsize_${dt}.log"
echo $log

touch input_tsize.txt
rm input_tsize.txt
for ntrain in `seq 7 13`;
do 
    mkdir results_tsize${ntrain}
    for trial in `seq 1 10`; 
    do 
        for model in pcc hybrid; 
        do 
            printf "${model}\t${trial}\t${ntrain}\n" >> input_tsize.txt
        done 
    done 

done 

#cat input_tsize.txt | parallel -v --dryrun --colsep '\t' "python test_with_trials.py {1} {2} --ntrain {3} &>> ${log}" 
cat input_tsize.txt | parallel --eta -j$CORES --colsep '\t' "python test_with_trials.py {1} {2} --ntrain {3} &>> ${log}" 