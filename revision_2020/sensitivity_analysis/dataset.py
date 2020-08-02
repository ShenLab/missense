from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

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


def get_train_valiate(input_config, gene_type):
    path = input_config['train'][gene_type]
    df = pd.read_csv(path)
    if gene_type == 'HIS':
        names = HIS_fea_names
    else:
        names = HS_fea_names
    x = df[names].values
    y = df['target'].values

    indices = np.arange(x.shape[0])
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        x, y, indices, test_size=0.2, random_state=42)

    return (X_train, y_train), (X_test, y_test)

def get_test(input_path, gene_type):
    df = pd.read_csv(input_path)
    if gene_type == 'HIS':
        names = HIS_fea_names
    else:
        names = HS_fea_names
    return df[names].values, df['var_id'].values
