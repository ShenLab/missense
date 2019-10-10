import numpy as np
import pandas as pd
import json
from sklearn.model_selection import StratifiedKFold as KFold
HS_fea_names = [
    "FATHMM_converted_rankscore", "Eigen-phred",
    "phastCons20way_mammalian_rankscore", "blosum62", "gc_content", "ASA",
    "lofz", "prec", "pli", "secondary_E", "secondary_H", "secondary_C",
    "complex_CORUM", "preppi_counts", "interface", "ubiquitination", "BioPlex",
    "pam250", "SUMO_score", "phospho_score", "domino"
]

HIS_fea_names = [
    "MutationAssessor_score_rankscore", "VEST3_rankscore",
    "Polyphen2_HDIV_rankscore", "Polyphen2_HVAR_rankscore",
    "SIFT_converted_rankscore", "PROVEAN_converted_rankscore",
    "FATHMM_converted_rankscore", "LRT_converted_rankscore",
    "Eigen-PC-raw_rankscore", "Eigen-phred", "Eigen-PC-phred",
    "phyloP20way_mammalian_rankscore", "GERP++_RS_rankscore",
    "SiPhy_29way_logOdds_rankscore", "phastCons100way_vertebrate_rankscore",
    "fathmm-MKL_coding_rankscore", "phyloP100way_vertebrate_rankscore",
    "MutationTaster_converted_rankscore", "phastCons20way_mammalian_rankscore",
    "blosum62", "pam250", "SUMO_score", "phospho_score", "lofz", "prec", "pli",
    "s_het_log", "secondary_E", "secondary_H", "complex_CORUM",
    "preppi_counts", "ASA", "secondary_C", "gc_content", "interface",
    "ubiquitination", "BioPlex", "obs_exp"
]


class Dataset:
    def __init__(self, input_config):
        if input_config['gene_type'] == 'HIS':
            self.fea_names = HIS_fea_names
        else:
            self.fea_names = HS_fea_names
        self.cv = input_config.get('cv', 10)
        if 'train' in input_config:
            data = pd.read_csv(input_config['train'], sep=',')
            y = data['target'].values
            X = np.zeros_like(y)
            skf = KFold(n_splits=self.cv, random_state=2018, shuffle=True)
            self.train_val_cv = []
            for train_index, val_index in skf.split(X, y):
                train = data.iloc[train_index, :]
                val = data.iloc[val_index, :]
                self.train_val_cv.append((train, val))

        if 'test' in input_config:
            self.test = {}
            for name, path in input_config['test'].items():
                self.test[name] = pd.read_csv(path, sep=',')

    def get_cv(self, mode, n):
        assert mode in ['train', 'val']
        assert n < self.cv
        if mode == 'train':
            data = self.train_val_cv[n][0]
        else:
            data = self.train_val_cv[n][1]
        data_X = data[self.fea_names].values
        data_Y = data['target'].values
        return data_X, data_Y

    def get_test(self, name):
        assert name in self.test
        data = self.test[name]
        data_X = data[self.fea_names].values
        data_Y = data['target'].values
        return data_X, data_Y

    def reload(self, name, path):
        self.test[name] = pd.read_csv(path, sep=',')
