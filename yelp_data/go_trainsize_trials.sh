#!/bin/zsh
base=2
for numtrain in `seq 7 13`;
do
    DIRNAME=train${numtrain}
    mkdir $DIRNAME
    for trial in `seq 1 10`; 
    do  
        mkdir ${DIRNAME}/trial${trial}
        num_to_select=$((${base}**${numtrain}))
        echo $num_to_select
        shuf train/train_all.json | head -n${num_to_select} > ${DIRNAME}/trial${trial}/train.json
        python getvocab.py ${numtrain} --trial ${trial}
    done 
done 