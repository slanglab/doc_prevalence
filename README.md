# doc_prevalence

Code to accompany the paper [Keith and O'Connor. "Uncertainty-aware generative models for inferring document class prevalence." EMNLP, 2018.](http://slanglab.cs.umass.edu/doc_prevalence/)

If you use this code please cite the paper: 
```
@inproceedings{keith18uncertainty,
author = {Keith, Katherine A. and O'Connor, Brendan}, 
title = {Uncertainty-aware generative models for inferring document class prevalence},
booktitle = {{EMNLP}},
year = 2018}
```

# Running LR-Implict
TODO 

# Code to replicate experiments in the paper

### Setting up Yelp dataset 
- `cd yelp_data`
- First download Yelp academic dataset challenge round 9 
https://www.yelp.com/dataset_challenge
- Open the .tar file 
- Then run 
`./go_preproc.sh`

This does the following pre-processing:
- tokenizes using NLK 
- unigrams 
- lowercase
- separates into classes (class=0 is stars=<3, class=1 is stars >3)
- prunes vocab so any vocab in the training data that is in <5 docs we don't use; also we use this same vocab size for the LSTM and map the pruned vocab to OOV symbols 

Output is `train/train_all.json` with dictionary keys

- "reivew_id" : review id from the original doc 

- "class" : class=0 is stars=<3, class=1 is stars >3

- "date" : original date 

- "toks" : dicionary of token counts 

#### train/test split

- First ignore all businesses that have less than 200 reviews
- Choose 500 test and 500 dev businesses by weighted random sampling
    - weighted random sampling is by the number of docs a business has

### 10-fold cross val; 10 re-samplings (Table 1, Fig 3) 
```
#(1) ASSEMBLE DATA 
yelp_data/go_maketrain_trials.sh #natural setting 
yelp_data/go_maketrain_trials_prop.sh #synthetic setting proportion of 0.1

#(2) TRAIN MODELS 
code/train_all/go_train_master.sh

#(3) EVALUATE MODELS ON THE TEST SET
code/eval/baselines.py 
code/eval/go_eval_master.sh 

#(4) ANALYZE RESULTS 
code/eval/analysis.py  
graphs/final_mae_plots.ipynb
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
graphs/train_prop10trains.ipynb
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
code/eval/analysis_train_size.py
graphs/trainsize.ipynb 
```

### Readme experiments
- First, you will need to download the [ReadMe R package](https://gking.harvard.edu/readme)
- Make sure you change your home path manually in `/readme_our_experiments/coderun_readme.R`
- Then run: 
```
code/readme_our_experiments/go_readme.sh
```
