#!/bin/zsh
#make all the different trials for the natural setting 
NTRAIN=2000

for i in `seq 10`;
do 
    echo $i
    DIR=train${NTRAIN}/trial${i}
    mkdir ${DIR}
    shuf train/train_all.json | head -n${NTRAIN} > ${DIR}/train.json
    python getvocab.py ${NTRAIN} --trial ${i}
done 