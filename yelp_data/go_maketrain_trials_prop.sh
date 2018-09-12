#!/bin/zsh
#make all the different trials
NTRAIN=2000
mkdir train${NTRAIN}_prop1

for i in `seq 10`;
do 
    echo $i
    DIR=train${NTRAIN}_prop1/trial${i}
    mkdir ${DIR}
    shuf train/train_all.json | head -n10000 > ${DIR}/train_toselect.json
    python prop_and_vocab.py ${i} 1
done 