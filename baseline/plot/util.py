import pandas as pd
import json
import numpy as np

dis_names = ['MVP_rank', 'RF_rank', 'DNN_rank', 'CADD_raw_rankscore']
density_names = ['MVP_rank', 'RF_rank', 'DNN_rank', 'CADD_raw_rankscore']

pr_thres_vec = [0.9, 0.75]
pr_config = {
    "MVP_rank": {
        "thres_vec": pr_thres_vec,
        "color": "red"
    },
    "RF_rank": {
        "thres_vec": pr_thres_vec,
        "color": "green"
    },
    "DNN_rank": {
        "thres_vec": pr_thres_vec,
        "color": "orange"
    },
    "CADD_raw_rankscore": {
        "thres_vec": pr_thres_vec,
        "color": "blue"
    },
}


def get_score(df, n):
    res = []
    for s in df[n]:
        try:
            if s is None:
                continue
            if type(s) == str:
                sa = s.split(',')
                sa = [float(ss) for ss in sa if ss != '.' and ss != '']
                if len(sa) == 0:
                    continue
                ss = float(sa[0])
            else:
                ss = float(s)
            if np.isnan(ss):
                continue
            res.append(ss)
        except:
            pass
    return np.array(res)
