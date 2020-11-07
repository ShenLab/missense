# MVP: predicting pathogenicity of missense variants by deep learning

This repository contains codes to train MVP model and predict missense variants pathogenicity. 


# code and tutorials: 
code/1_prepare_files_for_MVP.ipynb: Generate missense variants features for training and prediction for MVP models.

code/2_train_MVP_models.ipynb: Train MVP models with selected positives and negatives missense variants.

code/3_MVP_prediction.ipynb: Generate MVP raw scores for input variants. 

code/4_get_MVP_prediction_for_all_missense_variants.ipynb: Generate MVP raw scores for all rare missense variants, convert raw score to rank percentile scores.

code/models.py: functions used in MVP model.

# Precomputed MVP score (hg19)
The scores can be accessed through https://figshare.com/articles/dataset/Predicting_pathogenicity_of_missense_variants_by_deep_learning/13204118 or https://www.dropbox.com/s/d9we7gx42b7yatg/MVP_score_hg19.txt.bz2?dl=0.

# citation
This work is described in a preprint manuscript currently under peer-review:

Qi H*, Chen C*, Zhang H, Long JJ, Chung WK, Guan Y, Shen Y. (2018) MVP: predicting pathogenicity of missense variants by deep learning. bioRxiv, 259390 
https://www.biorxiv.org/content/early/2018/04/02/259390
