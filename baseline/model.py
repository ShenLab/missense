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
from sklearn.externals import joblib

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Input, Dense, BatchNormalization
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.regularizers import l2


class FCNN:
    def __init__(self, feature_dim):
        #build models
        model = Sequential()
        #model.add(Input(shape=(feature_dim,)))
        ef = 0.0

        model.add(Dense(units=64, activation='relu',
                        kernel_regularizer=l2(ef)))
        model.add(Dense(units=32, activation='relu',
                        kernel_regularizer=l2(ef)))
        model.add(Dense(units=1, activation='sigmoid'))
        adam = Adam(0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile(loss='binary_crossentropy', optimizer=adam)
        self.model = model

    def fit(self, X, y, val_X, val_y):
        print('train shape', X.shape, y.shape)
        es = EarlyStopping(monitor='val_loss',
                           mode='min',
                           verbose=1,
                           patience=5,
                           restore_best_weights=True)
        self.model.fit(X,
                       y,
                       validation_data=(val_X, val_y),
                       verbose=1,
                       batch_size=32,
                       epochs=128,
                       callbacks=[es])

    def predict_proba(self, X):
        proba = self.model.predict(X)
        pp = 1.0 - proba
        proba = np.concatenate([pp, proba], axis=1)
        print('proba', proba.shape)
        return proba

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = load_model(path)


class Model:
    def __init__(self, dataset, model_name, random_state=2018):
        self.dataset = dataset
        self.models = []
        self.random_state = random_state
        self.feature_dim = len(dataset.fea_names)
        assert model_name in ['RF', 'DNN']
        self.model_name = model_name

    def _train(self, train_X, train_Y, val_X, val_Y, random_state=2018):
        if self.model_name == 'RF':
            model = RandomForestClassifier(n_estimators=256,
                                           max_features='auto',
                                           criterion='gini',
                                           max_depth=12,
                                           min_samples_split=10,
                                           random_state=random_state)
        if self.model_name == 'DNN':
            model = FCNN(train_X.shape[1])

        model.fit(train_X, train_Y, val_X, val_Y)
        return model

    def train(self):
        for n in range(self.dataset.cv):
            print('train', n, self.model_name)
            train_X, train_Y = self.dataset.get_cv('train', n)
            val_X, val_Y = self.dataset.get_cv('val', n)
            print('data shape', train_X.shape, train_Y.shape)
            model = self._train(train_X,
                                train_Y,
                                val_X,
                                val_Y,
                                random_state=self.random_state + n)
            self.models.append(model)
        return

    def save_model(self, model_dir):
        for idx, m in enumerate(self.models):
            if self.model_name == 'RF':
                joblib.dump(m, '{}/model{}.pkl'.format(model_dir, idx))
            if self.model_name == 'DNN':
                m.save_model('{}/model{}.h5'.format(model_dir, idx))

    def load_model(self, model_dir, cv):
        self.models = []
        for i in range(cv):
            if self.model_name == 'RF':
                m = joblib.load('{}/model{}.pkl'.format(model_dir, i))
            if self.model_name == 'DNN':
                m = FCNN(self.feature_dim)
                m.load_model('{}/model{}.h5'.format(model_dir, i))

            self.models.append(m)

    def _predict(self, model, test_X):
        pred = model.predict_proba(test_X)
        return pred[:, 1]

    def predict(self, name):
        test_X, test_Y = self.dataset.get_test(name)
        preds = []
        for m in self.models:
            pred = self._predict(m, test_X)
            preds.append(pred)
        preds = np.array(preds)
        return preds, test_Y

    def evaluate_auc(self, name):
        preds, target = self.predict(name)
        for i in range(preds.shape[0]):
            pred = preds[i, :]
            auROC = sk.metrics.roc_auc_score(target, pred)
            print('name={} cv= {} auROC= {:.4f}'.format(name, i, auROC))
        avg = np.mean(preds, axis=0)
        auROC = sk.metrics.roc_auc_score(target, avg)
        print('name={} cv= avg auROC= {:.4f}'.format(name, auROC))

    def evaluate_denovo(self):
        preds_asd, target = self.predict('asd')
        preds_chd, target = self.predict('chd')
        preds_control, target = self.predict('control')
        for i in range(preds_asd.shape[0]):
            asd = preds_asd[i, :]
            chd = preds_chd[i, :]
            control = preds_control[i, :]
            chd_u = mannwhitneyu(chd, control, alternative='two-sided')
            asd_u = mannwhitneyu(asd, control, alternative='two-sided')
            print('cv={}'.format(i))
            print('chd= {} asd= {} control= {}'.format(chd.shape, asd.shape,
                                                       control.shape))
            print("denovo MannWhitneyU chd_u= {}".format(chd_u))
            print("denovo MannWhitneyU asd_u= {}".format(asd_u))
        chd = np.mean(preds_chd, axis=0)
        asd = np.mean(preds_asd, axis=0)
        control = np.mean(preds_control, axis=0)
        chd_u = mannwhitneyu(chd, control, alternative='two-sided')
        asd_u = mannwhitneyu(asd, control, alternative='two-sided')
        print('chd= {} asd= {} control= {}'.format(chd.shape, asd.shape,
                                                   control.shape))
        print("denovo MannWhitneyU mean chd_u= {}".format(chd_u))
        print("denovo MannWhitneyU mean asd_u= {}".format(asd_u))
