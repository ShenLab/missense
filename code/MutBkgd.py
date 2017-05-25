import sys



class MutationBackground(object):
    """
    MutationBackground class used to calcuate mutation background
    """

    def __init__(self, fname):
        """Return a MutationBackground object whose rate is a dict with each gene
        and its mutation background rate."""
        self.mut_type = {'p_misense', 'p_mcap',  'p_mcap_0.05', 'p_metaSVM', 'p_metalr', 
                         'p_cadd10', 'p_cadd15', 'p_cadd20', 'p_cadd25', 'p_cadd30', 'p_cadd35',
                         'p_pp2Hvar', 'p_fahmm', 'peigen_pred10',   'peigen_pred15',   'peigen_pc10', 'peigen_pc15',
                         '0.05',   '0.1',   '0.15',  '0.2',   '0.25',   '0.3',   '0.35',   '0.4',   '0.45',   '0.5',
                         '0.55',   '0.6',   '0.65',
                         '0.7',   '0.75',   '0.8',   '0.85',   '0.9',   '0.56'}

        self._init_rate_Na(fname)

    def _convert(self, mutation_type):
        ''' used for convert some cols name difference from input file
        '''
        mutation_type_syn = {'p_misense': 'all_missense', 'p_mcap': 'M_CAP>0.025',  'p_mcap_0.05': 'M_CAP>0.05',
                             'p_metaSVM': 'MetaSVM>0', 'p_metalr': 'MetaLR>0', 
                             'p_cadd10': 'cadd10', 'p_cadd15': 'cadd15', 'p_cadd20': 'cadd20', 'p_cadd25': 'cadd25',
                             'p_cadd30': 'cadd30',    'p_cadd35': 'cadd35',
                             'p_pp2Hvar': 'PP2-HVAR',   'p_fahmm': 'FATHMM', 
                             'peigen_pred10': 'eigen_pred10',   'peigen_pred15': 'eigen_pred15',
                             'peigen_pc10': 'eigen_pc_pred10', 'peigen_pc15': 'eigen_pc_pred15',
                             '0.05': 'cnn_0.05',   '0.1': 'cnn_0.1', '0.15': 'cnn_0.15',  '0.2': 'cnn_0.2',   '0.25': 'cnn_0.25',
                             '0.3': 'cnn_0.3',   '0.35':  'cnn_0.35',   '0.4': 'cnn_0.4',   '0.45': 'cnn_0.45', '0.5': 'cnn_0.5', 
                             '0.55': 'cnn_0.55', '0.6': 'cnn_0.6',   '0.65': 'cnn_0.65',
                             '0.7': 'cnn_0.7',   '0.75': 'cnn_0.75',   '0.8': 'cnn_0.8',
                             '0.85': 'cnn_0.85',   '0.9': 'cnn_0.9',   '0.56': 'cnn_best_0.56'}

        return mutation_type_syn.get(mutation_type, mutation_type)

    def _init_rate_Na(self, fname):
        """inti mutation rate using Na's format, used the longest exon as final reslut."""
        self.rate = {}
        with open(fname) as f:
            head = f.readline().strip().split()
            for line in f:
                lst = line.strip().split()
                info = dict(zip(head, lst))
                gene = info['Gene']
                rate = {}
                for mut_type in self.mut_type:
                    mut_type_converted = self._convert(mut_type)
                    rate[mut_type_converted] = float(info[mut_type])
                if gene not in self.rate:
                    self.rate[gene] = rate
                else:
                    print 'something wrong:', gene

    def expectation(self, geneset, mut_type, verbose=True):
        """Return the mutation background given a mutation rate dict."""
        exp = 0
        cur_mut_type = set(self.rate[self.rate.keys()[0]].keys())
        if mut_type in cur_mut_type:
            for gene in geneset:
                if gene in self.rate:
                    exp += float(self.rate[gene][mut_type]) * 2
        else:
            if verbose:
                print 'do not have rate for {}'.format(mut_type)
        return exp
