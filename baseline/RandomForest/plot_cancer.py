import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

HIS = pd.read_csv('./final_HIS/pred/cancer_HIS.csv', sep=',')
HS = pd.read_csv('./final_HS/pred/cancer_HS.csv', sep=',')

HIS = HIS[['cnn_prob', 'RF', 'target']]
HS = HS[['cnn_prob', 'RF', 'target']]

All = pd.concat([HIS, HS], axis=0)

print(HIS.shape, HS.shape, All.shape)

for df, df_n in zip([HIS, HS, All], ['cancer_HIS', 'cancer_HS', 'cancer_All']):
    fig, ax = plt.subplots()
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    for alg in ['cnn_prob', 'RF']:
        score = df[alg].values
        target = df['target'].values
        fpr, tpr, _ = roc_curve(target, score)
        roc_auc = auc(fpr, tpr)
        sn = 'MVP' if alg == 'cnn_prob' else 'RF'
        color = 'black' if alg == 'cnn_prob' else 'red'
        ax.plot(fpr, tpr, color=color, label=f'{sn} ({roc_auc:.2f})')
        ax.set_aspect('equal')
    ax.legend()
    fig.savefig(f'{df_n}.pdf', bbox_inches='tight')
    plt.close(fig)
