import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

DNN_HIS = pd.read_csv('../DNN/final_HIS/pred/cancer_HIS_rank.csv',
                      sep=',')[['cnn_prob', 'var_id', 'DNN_rank', 'target']]
DNN_HS = pd.read_csv('../DNN/final_HS/pred/cancer_HS_rank.csv',
                     sep=',')[['cnn_prob', 'var_id', 'DNN_rank', 'target']]

RF_HIS = pd.read_csv('../RF/final_HIS/pred/cancer_HIS.csv',
                     sep=',')[['var_id', 'RF']]
RF_HS = pd.read_csv('../RF/final_HS/pred/cancer_HS.csv',
                    sep=',')[['var_id', 'RF']]

HIS = pd.merge(DNN_HIS, RF_HIS, on='var_id')
HS = pd.merge(DNN_HS, RF_HS, on='var_id')
All = pd.concat([HIS, HS], axis=0)

print(HIS.shape, HS.shape, All.shape)

for df, df_n in zip([HIS, HS, All], ['cancer_HIS', 'cancer_HS', 'cancer_All']):
    fig, ax = plt.subplots()
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    for alg in ['cnn_prob', 'RF', 'DNN_rank']:
        score = df[alg].values
        target = df['target'].values
        fpr, tpr, _ = roc_curve(target, score)
        roc_auc = auc(fpr, tpr)

        if alg == 'cnn_prob':
            sn = 'MVP'
            color = 'black'
        if alg == 'DNN_rank':
            sn = 'FCNN'
            color = 'red'
        if alg == 'RF':
            sn = 'RF'
            color = 'blue'
        ax.plot(fpr, tpr, color=color, label=f'{sn} ({roc_auc:.2f})')
        ax.set_aspect('equal')
    ax.legend()
    fig.savefig(f'./figures/{df_n}.pdf', bbox_inches='tight')
    plt.close(fig)
