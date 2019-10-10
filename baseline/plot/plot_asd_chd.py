import pandas as pd
import argparse
import numpy as np
from scipy.stats import mannwhitneyu
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from scipy import stats
import seaborn as sns
import util
from scipy.stats import binom_test

asd_total = 3953
chd_total = 2645
control_total = 1911

asd_case_control_ratio = 1.0 * asd_total / (asd_total + control_total)
chd_case_control_ratio = 1.0 * chd_total / (chd_total + control_total)

asd_alpha = 1.04859
chd_alpha = 1.02691

print(chd_alpha, asd_alpha)


def _get_scores(asd, chd, control, n):
    return util.get_score(asd, n),\
            util.get_score(chd, n),\
            util.get_score(control, n)


def _plot_density(asd_score, chd_score, control_score, save_path, n):
    chd_u = mannwhitneyu(chd_score, control_score, alternative='two-sided')
    asd_u = mannwhitneyu(asd_score, control_score, alternative='two-sided')

    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.kdeplot(asd_score, ax=ax, color='purple', label='ASD')
    sns.kdeplot(chd_score, ax=ax, color='red', label='CHD')
    sns.kdeplot(control_score, ax=ax, color='blue', label='Control')
    legend = plt.legend(loc='lower left')
    #,
    #'ASD vs controls: {:.1e}'.format(asd_u.pvalue)
    ax.text(
        0.1, 1.4,
        'CHD vs controls: p={:.1e}\nASD vs controsl: p={:.1e}'.format(
            chd_u.pvalue, asd_u.pvalue))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.5)
    ax.set_yticks(np.arange(0, 1.6, 0.5))
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(linestyle='--', linewidth=0.5)
    ax.set_xlabel('Rank score')
    ax.set_ylabel('Density')
    ax.set_title(n, fontsize='large', fontweight='bold')
    ax.add_artist(legend)
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    return


def _calc_precision(case_scores, control_scores, case_total, control_total,
                    thres, alpha):
    case_count = np.sum(case_scores > thres)
    control_count = np.sum(control_scores > thres)
    control_count = max(control_count, 1)
    case_rate = 1.0 * case_count / case_total
    control_rate = 1.0 * control_count / control_total
    enrichment = case_rate / control_rate / alpha
    #enrichment = max(enrichment, 1.0)
    tpr = 1.0 - 1.0 / enrichment
    num = case_count * tpr
    print(thres, case_count, tpr, num, case_rate, control_rate, enrichment)
    return tpr, num, case_count, control_count


def _calc_precision_vec(case_scores, control_scores, case_total, control_total,
                        thres_vec, alpha):
    p_vec, num_vec, pvalue_vec = [], [], []
    for thres in thres_vec:
        estimated_tpr, estimated_num, case_count, control_count = _calc_precision(
            case_scores, control_scores, case_total, control_total, thres,
            alpha)
        p_vec.append(estimated_tpr)
        num_vec.append(estimated_num)
        pvalue = binom_test(case_count, case_count + control_count,
                            1.0 * case_total / (case_total + control_total))
        pvalue_vec.append(pvalue)
    return np.array(p_vec), np.array(num_vec), np.array(pvalue_vec)


def marker_size_pvalue(pvalue):
    if pvalue > 0.5:
        return 0
    return -1.0 * np.log(pvalue)


def _plot_pr(config, cases_df, controls_df, case_total, control_total, alpha,
             fig_path):
    fig, ax = plt.subplots()
    #ax.set_ylim(0, 0.5)
    ax.set_yticks(np.arange(0, 0.51, 0.1))
    for name, info in config.items():
        case_scores = util.get_score(cases_df, name)
        control_scores = util.get_score(controls_df, name)
        print(name, case_scores.shape, control_scores.shape)

        estimated_tpr, estimated_num, pvalues = _calc_precision_vec(
            case_scores, control_scores, case_total, control_total,
            info['thres_vec'], alpha)
        print(estimated_tpr)
        print(estimated_num)

        ax.scatter(estimated_num, estimated_tpr, color=info['color'])
        if name == 'cnn_prob':
            name = 'MVP'
        if 'DNN' in name:
            name = 'FCNN'
        c = info['color']
        marker = '.'
        for num_, tpr_, pvalue_ in zip(estimated_num, estimated_tpr, pvalues):
            markersize = marker_size_pvalue(pvalue_)
            ax.plot(num_, tpr_, markersize=markersize, marker=marker, color=c)
        ax.plot(estimated_num,
                estimated_tpr,
                linestyle='-',
                color=info['color'],
                label=name.split('_')[0])

        for thres_, num_, tpr_ in zip(info['thres_vec'], estimated_num,
                                      estimated_tpr):
            ax.annotate("{}".format(thres_),
                        xy=(num_, tpr_),
                        ha='left',
                        va='bottom')

        if name == 'MVP_rank':
            estimated_tpr, estimated_num, pvalues = _calc_precision_vec(
                case_scores, control_scores, case_total, control_total, [-1.0],
                alpha)
            c = 'purple'
            ax.plot(estimated_num[0],
                    estimated_tpr[0],
                    color=c,
                    marker='.',
                    markersize=marker_size_pvalue(pvalues[0]))
            #ax.plot(estimated_num,
            #        estimated_tpr,
            #        linestyle='-',
            #        color=c,
            #        label=name.split('_')[0])
            ax.annotate("All Mis",
                        xy=(estimated_num[0], estimated_tpr[0]),
                        ha='left',
                        va='bottom')

    ax.set_ylabel('Estimated Positive Predictive Value', weight='normal')
    ax.set_xlabel('Estimated number of risk variants', weight='normal')
    lgnd = ax.legend(loc='upper right')
    # plot pvalue legend
    ls = []
    for p in [10**-8, 10**-6, 10**-4, 10**-2]:
        l, = ax.plot([], [],
                     'o',
                     marker='.',
                     markersize=-np.log(p),
                     color='black')
        ls.append(l)
    labels = ["10E-8", "10E-6", "10E-4", "10E-2"]
    leg = ax.legend(ls,
                    labels,
                    numpoints=1,
                    ncol=4,
                    frameon=False,
                    loc='lower center',
                    handlelength=2,
                    borderpad=0,
                    handletextpad=1,
                    title='p value')
    leg.get_title().set_fontsize('8')  # legend 'Title' fontsize

    #plt.setp(plt.gca().get_legend().get_texts(),
    #         fontsize='20')  # legend 'list' fontsize
    plt.gca().add_artist(lgnd)
    #ax.set_xlim([50, 300])
    #plt.xticks(fontsize = 28)
    #plt.yticks(fontsize = 28)
    #plt.legend(loc="upper right")
    #ax.legend(loc="best")
    plt.savefig(fig_path, format='pdf', bbox_inches='tight')
    plt.close(fig)


def plot_density(asd, chd, control):
    for n in util.density_names:
        asd_, _, control_ = _get_scores(asd, chd, control, n)
        print(n, asd_.shape, _.shape, control_.shape)
        if n == 'cnn_prob':
            n = 'MVP'
        if 'DNN' in n:
            n = 'FCNN'
        save_path = './figures/{}_{}_ASD_CHD_density.pdf'.format(
            args.prefix,
            n.split('_')[0])
        _plot_density(asd_, _, control_, save_path, n.split('_')[0])


def plot_pr(asd, chd, control):
    save_path = f'./figures/{args.prefix}_ASD_PR.pdf'
    _plot_pr(util.pr_config, asd, control, asd_total, control_total, asd_alpha,
             save_path)

    save_path = f'./figures/{args.prefix}_CHD_PR.pdf'
    _plot_pr(util.pr_config, chd, control, chd_total, control_total, chd_alpha,
             save_path)


def main(args):
    asd = pd.read_csv(args.asd, sep='\t')
    chd = pd.read_csv(args.chd, sep='\t')
    control = pd.read_csv(args.control, sep='\t')
    print(chd.shape[0])
    print(asd.shape[0])
    print(control.shape[0])
    plot_density(asd, chd, control)
    plot_pr(asd, chd, control)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--asd', type=str)
    parser.add_argument('--chd', type=str)
    parser.add_argument('--control', type=str)
    parser.add_argument('--prefix', type=str, default='All')
    args = parser.parse_args()
    main(args)
