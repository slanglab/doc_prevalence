#!/bin/zsh
#make sure you've downloaded the yelp data
mkdir train_nopreproc test_nopreproc dev_nopreproc
mkdir train test dev 

#separate the data into train, test, and dev by business 
python bybusiness.py

CORES=10
#too many files so will have to run the command
find train_nopreproc/ -type f | parallel --eta -j$CORES "python preproc.py {}"

#pre-process test and dev businesses  
ls -1 test_nopreproc/* | parallel -v --dryrun "python preproc.py {}"
ls -1 test_nopreproc/* | parallel --eta -j$CORES "python preproc.py {}"

ls -1 dev_nopreproc/* | parallel -v --dryrun "python preproc.py {}"
ls -1 dev_nopreproc/* | parallel --eta -j$CORES "python preproc.py {}"

./fullcat.sh > train/train_all.json

