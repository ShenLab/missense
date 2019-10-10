import pandas as pd
import argparse
import numpy as np
from scipy.stats import percentileofscore as percentile

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--bg_path', type=str, required=True)
    parser.add_argument('--alg_name', type=str, required=True)
    args = parser.parse_args()
    bg = np.loadtxt(args.bg_path)
    print bg.shape
    df = pd.read_csv(args.input_path)
    alg_name = args.alg_name

    def get_rank(x):
        return percentile(bg, x) / 100.0

    df["{}_rank".format(alg_name)] = df[alg_name].apply(get_rank)

    df.to_csv(args.input_path.split('.csv')[0] + '_rank.csv',
              index=False,
              sep=',')
