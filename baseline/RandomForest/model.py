from scipy.stats import mannwhitneyu
import random
import argparse
import numpy as np
import pandas as pd
import json
import sklearn as sk
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
import cPickle
import lightgbm as lgb
from sklearn.externals import joblib
        
class ModelRF:
    def __init__(self, dataset, random_state=2018):
        self.dataset = dataset
        self.models = []
        self.random_state=random_state
    
    def _train(self, train_X, train_Y, random_state=2018):
        model = RandomForestClassifier(
                n_estimators = 256,
                max_features = 'auto',
                criterion = 'gini',
                max_depth = 12,
                min_samples_split = 10,
                random_state = random_state)
        model.fit(train_X, train_Y)
        return model

    def train(self):
        for n in range(self.dataset.cv):
            train_X, train_Y = self.dataset.get_cv('train', n)
            model=self._train(train_X, train_Y, random_state=self.random_state+n)
            self.models.append(model)
        return

    def save_model(self, model_dir):
        for idx, m in enumerate(self.models):
            joblib.dump(m, '{}/model{}.pkl'.format(model_dir, idx))

    def load_model(self, model_dir, cv):
        self.models=[]
        for i in range(cv):
            m = joblib.load('{}/model{}.pkl'.format(model_dir, i)) 
            self.models.append(m)
    
    def _predict(self, model, test_X):
        pred = model.predict_proba(test_X)
        return pred[:, 1]
        
    def predict(self, name):
        test_X, test_Y = self.dataset.get_test(name)
        preds=[]
        for m in self.models:
            pred=self._predict(m, test_X)
            preds.append(pred)
        preds=np.array(preds)
        return preds, test_Y
    
    def evaluate_auc(self, name):
        preds, target=self.predict(name)
        for i in range(preds.shape[0]):
            pred=preds[i,:]
            auROC=sk.metrics.roc_auc_score(target, pred)
            print 'name={} cv= {} auROC= {:.4f}'.format(name, i, auROC)
        avg=np.mean(preds, axis=0)
        auROC=sk.metrics.roc_auc_score(target, avg)
        print 'name={} cv= avg auROC= {:.4f}'.format(name, auROC)
        
    def evaluate_denovo(self):
        preds_asd, target=self.predict('asd')
        preds_chd, target=self.predict('chd')
        preds_control, target=self.predict('control')
        for i in range(preds_asd.shape[0]):
            asd=preds_asd[i,:]
            chd=preds_chd[i,:]
            control=preds_control[i,:]
            chd_u=mannwhitneyu(chd, control, alternative='two-sided')
            asd_u=mannwhitneyu(asd, control, alternative='two-sided')
            print 'cv={}'.format(i)
            print('chd= {} asd= {} control= {}'.format(chd.shape, asd.shape, control.shape))
            print("denovo MannWhitneyU chd_u= {}".format(chd_u))
            print("denovo MannWhitneyU asd_u= {}".format(asd_u))
        chd=np.mean(preds_chd, axis=0)
        asd=np.mean(preds_asd, axis=0)
        control=np.mean(preds_control, axis=0)
        chd_u=mannwhitneyu(chd, control, alternative='two-sided')
        asd_u=mannwhitneyu(asd, control, alternative='two-sided')
        print('chd= {} asd= {} control= {}'.format(chd.shape, asd.shape, control.shape))
        print("denovo MannWhitneyU mean chd_u= {}".format(chd_u))
        print("denovo MannWhitneyU mean asd_u= {}".format(asd_u))
