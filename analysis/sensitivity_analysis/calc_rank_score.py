import pandas as pd
import argparse
import numpy as np
from scipy.stats import percentileofscore as percentile

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--sample_path', type=str, required=True)
    args=parser.parse_args()
    
    sample=pd.read_csv(args.sample_path)['pred'].values
    df=pd.read_csv(args.input_path)

    def get_rank(x):
        return percentile(sample, x)/100.0

    df['MVP_rank']=df['pred'].apply(get_rank)

    df.to_csv(args.input_path.split('.csv')[0] + '.rank.csv', index=False, sep=',')
