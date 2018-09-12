#!/bin/zsh
#(1) transform data to csv
rm -rf test train 
mkdir train test
mkdir train/nat train/prop1 
mkdir test/nat test/prop1

for trial in `seq 1 10`;
do 
    mkdir train/nat/trial${trial}
    mkdir train/prop1/trial${trial}
    mkdir test/nat/trial${trial}
    mkdir test/prop1/trial${trial}
done

python data_to_csv.py 

# #(2) run readme 
for trial in `seq 1 10`;
do 
    mkdir -p results/nat/trial${trial}
    mkdir -p results/prop1/trial${trial} 
done

touch input_par.txt
rm input_par.txt
for setting in nat prop1;
do
    for trial in `seq 1 10`;
    do
        for testgroup in `seq 1 500`;
        do
            printf "${setting}\t${trial}\t${testgroup}\n" >> input_par.txt
        done 
    done 
done

CORES=30

#cat input_par.txt | parallel -v --dryrun --colsep '\t' "Rscript run_readme.R {1} {2} {3}"
cat input_par.txt | parallel --eta -j${CORES} --colsep '\t' "Rscript run_readme.R {1} {2} {3}"

#(3) anlyze the results
mkdir -p save_eval/nat save_eval/prop1
python analyze_readme.py 




