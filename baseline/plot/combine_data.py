import pandas as pd


def combine(name):
    DNN_HIS = pd.read_csv(
        f'../DNN/final_HIS/pred/{name}_HIS_rank.csv',
        sep=',')[['MVP_rank', 'var_id', 'DNN_rank', 'CADD_raw_rankscore']]
    DNN_HS = pd.read_csv(f'../DNN/final_HS/pred/{name}_HS_rank.csv', sep=',')[[
        'MVP_rank', 'var_id', 'DNN_rank', 'CADD_raw_rankscore'
    ]]

    RF_HIS = pd.read_csv(f'../RF/final_HIS/pred/{name}_HIS_rank.csv',
                         sep=',')[['var_id', 'RF_rank']]
    RF_HS = pd.read_csv(f'../RF/final_HS/pred/{name}_HS_rank.csv',
                        sep=',')[['var_id', 'RF_rank']]

    HIS = pd.merge(DNN_HIS, RF_HIS, on='var_id')
    HS = pd.merge(DNN_HS, RF_HS, on='var_id')
    All = pd.concat([HIS, HS], axis=0)
    HIS.to_csv(f'{name}_HIS.csv', sep='\t', index=False)
    HS.to_csv(f'{name}_HS.csv', sep='\t', index=False)
    All.to_csv(f'{name}_All.csv', sep='\t', index=False)


combine('asd')
combine('chd')
combine('control')
