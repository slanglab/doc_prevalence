#!/bin/zsh
#make all the different trials for all the different training proprotions
NTRAIN=2000
CORES=10

shuf train/train_all.json | head -n100000 > train/train_toselect100k.json

touch input_train_prop.txt
for prop in `seq 2 10`; #starts at 2 since already made prop1 for other experiments 
do 
    mkdir train${NTRAIN}_prop${prop}
    for i in `seq 10`;
    do
        printf "${prop}\t${i}\n"
        printf "${prop}\t${i}\n" >> input_train_prop.txt
        DIR=train${NTRAIN}_prop${prop}/trial${i}
        mkdir ${DIR}
        shuf train/train_all.json | head -n10000 > ${DIR}/train_toselect.json
    done
done 

cat input_train_prop.txt | parallel -v --dryrun --colsep '\t' "python prop_and_vocab.py {1} {2}"
cat input_train_prop.txt | parallel --eta -j$CORES --colsep '\t' "python prop_and_vocab.py {1} {2}"

