# doc_prevalence
Code to accompany the paper ["Uncertainty-aware generative models for inferring document class prevalence." EMNLP, 2018.](http://slanglab.cs.umass.edu/doc_prevalence/)

## Cite 
If you use this code please cite 
```
@inproceedings{keith18uncertainty,
author = {Keith, Katherine A. and O'Connor, Brendan}, 
title = {Uncertainty-aware generative models for inferring document class prevalence},
booktitle = {{EMNLP}},
year = 2018}
```

## Running LR-Implict 

## Code to replicate experiments in the paper 

### 10-fold cross val; 10 re-samplings (Table 1, Fig 3) 
```
#(1) ASSEMBLE DATA 
yelp_data/go_maketrain_trials.sh
yelp_data/go_maketrain_trials_prop.sh

#(2) TRAIN MODELS 
code/train_all/go_train_master.sh

#(3) EVALUATE MODELS ON THE TEST SET
code/eval/go_eval_master.sh 
code/eval/baselines.py 

#(4) DO ANALYSIS ON THE RESULTS 
code/eval/analysis.py  
graphs/5-12_final_mae_plots.ipynb

```


### Training proportion experiments (10 resamplings of training data) (Fig 5a)
```
#(1) ASSEMBLE DATA 
yelp_data/go_train_prop_all.sh

#(2) TRAIN MODELS 
code/train_all/go_training_prop_all.sh

#(3) EVAL MODELS ON THE TEST SET 
code/eval/go_eval_train_prop_all.sh

#(4) ANALYZE RESULTS
code/eval/analysis_train_prop.py

```


### Training size experiments (10 resamplings of training data) (Fig 5b)
```
#(1) ASSEMBLE DATA 
yelp_data/go_trainsize_trials.sh

#(2) TRAIN
code/train_all/go_trainsize_trials.sh

#(3) EVAL
code/eval/go_trainsize_all_trials.sh

#(4) ANALYSIS 
```
