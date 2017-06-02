import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import datetime
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Lambda, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils
from keras import backend as K
from IPython.display import SVG, display
#from keras.utils.visualize_util import model_to_dot, plot
from keras.utils.vis_utils import plot_model
import numpy

import pandas as pd

from random import shuffle
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import sys

# fix random seed for reproducibility
seed = 7
seed = 1337
numpy.random.seed(seed)


class CNN_Model(object):

    """"simple cnn models
    https://github.com/fchollet/keras/blob/master/examples/imdb_cnn.py
    """

    def __init__(self, input_shape=(20, 1, 1), weights_path=None, name='cnn_model',
                 train_flag=True, nb_epoch=50, batch_size=64, verbose=0,
                 fname='../data/input_data.csv',
                 f_out='../data/output_data.csv'):

        self.input_shape = input_shape
        self.name = name
        self.weights_path = weights_path
        self.fname = fname
        self.f_out = f_out
        self.train_flag = train_flag
        self.min_val_loss = sys.float_info.max
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.verbose = verbose
        self.run_id = datetime.datetime.now().strftime('%Y%m%d-%H.%M.%S')
        self.exclude_cols = {'target', 'CADD_phred', 'xEigen-phred', 'Eigen-PC-phred',
                             'Eigen-PC-raw_rankscore', 'MetaSVM_rankscore',
                             'MetaLR_rankscore', 'M-CAP_rankscore', 'DANN_rankscore',
                             'CADD_raw_rankscore', 'Polyphen2_HVAR_rankscore',
                             'MutationTaster_converted_rankscore',
                             '#chr', 'pos(1-based)',  'hg19_chr', 'hg19_pos(1-based)',
                             'ref', 'alt', 'category',
                             'source', 'INFO', 'disease', 'genename',
                             'pli', 'lofz', 'prec', 
                             'x1000Gp3_AF', 'xExAC_AF', 
                             's_het', 'xs_het_log', 'xgc_content', 
                             'xFATHMM_converted_rankscore', 'xfathmm-MKL_coding_rankscore',
                             'xpreppi_counts', 'xubiquitination'}

        # self.exclude_cols = {'target', 'CADD_phred', 'Eigen-phred', 'Eigen-PC-phred',
        #                      'Eigen-PC-raw_rankscore', 'MetaSVM_rankscore',
        #                      'MetaLR_rankscore', 'M-CAP_rankscore', 'DANN_rankscore',
        #                      'CADD_raw_rankscore', 'Polyphen2_HVAR_rankscore',
        #                      'MutationTaster_converted_rankscore',
        #                      '#chr', 'pos(1-based)',  'ref', 'alt', 'category',
        #                      'source', 'INFO', 'disease', 'genename',
        #                      'xpli', 'xlofz', 
        #                      'x1000Gp3_AF', 'xExAC_AF',
        #                      'xFATHMM_converted_rankscore', 'xfathmm-MKL_coding_rankscore',
        #                      'xpreppi_counts', 'xubiquitination'}
    def _load_data(self, sub_sample=False):
        '''load data are not in exclude_cols into self.X_pred, 
           feathers to self.X_train
           target in self.y is exist
        '''
        print('Loading training data...')
        self.data = pd.read_csv(self.fname)

        if sub_sample:
            pos = self.data[self.data['target']==1]
            neg = self.data[self.data['target']==0]
            
            if pos.shape[0] < neg.shape[0]:
                neg = neg.sample(pos.shape[0])
            else:
                pos = pos.sample(neg.shape[0])
            self.data = pd.concat([pos, neg], ignore_index=True)
            print pos.shape, neg.shape
        cols = [col for col in self.data.columns if col not in self.exclude_cols]
        self.cols = cols
        
        print '{} cols used: {}'.format(len(cols), cols)
        self.input_shape = (len(cols), 1, 1)

        # set attributes of all data used for prediction
        self.X_pred = self.data[cols].values
        self.X_pred = self.X_pred.reshape(
            self.X_pred.shape[0], self.X_pred.shape[1], 1, 1)

        if 'target' in self.data.columns:
            self.y = self.data['target']
            self.X_train = self.X_pred
            self.y_train = self.y

    def _train_test_split(self):
        """80/20 split, 80% for training, 20% for validation
        """
        indices = numpy.arange(self.X_pred.shape[0])
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            self.X_pred, self.y, indices, test_size=0.2, random_state=42)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.idx_train = idx_train
        self.idx_test = idx_test
        print('X_train shape:', self.X_train.shape)
        print(self.X_train.shape[0], 'train samples')
        print(self.X_test.shape[0], 'test samples')

    def _init_model(self, verbose):
        # number of convolutional filters to use
        nb_filters = 32
        # size of pooling area for max pooling
        pool_size = (2, 1)
        # convolution kernel size
        kernel_size = (3, 1)

        input_ = Input(shape=self.input_shape, name='input')
        conv1 = Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                              border_mode='valid', dim_ordering='tf',
                              activation='relu', init='glorot_uniform', name='conv1')(input_)
        conv2 = Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                              border_mode='valid', dim_ordering='tf',
                              activation='relu', init='glorot_uniform', name='conv2')(conv1)
        maxpool1 = MaxPooling2D(pool_size=pool_size,
                                dim_ordering='tf', name='maxpool1')(conv2)
        dropout1 = Dropout(0.25, name='dropout1')(maxpool1)
        flatten = Flatten(name='flatten')(dropout1)
        fc1 = Dense(128, name='fc1')(flatten)
        act1 = Activation('relu', name='act1')(fc1)
        dropout2 = Dropout(0.25, name='dropout2')(act1)
        dense1 = Dense(1, name='dense1')(dropout2)
        act2 = Activation('sigmoid', name='act2')(dense1)

        self.model = Model(input=input_, output=act2)
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

        if self.weights_path:
            self.model.load_weights(self.weights_path)

        if verbose:
            # model summary and save arch
            print self.model.summary()
            outname = '_'.join(['../models/'+self.name, str(self.input_shape[0]), self.run_id,  'cols.png'])
            plot_model(self.model, show_shapes=True, to_file=outname)
            col_name = '_'.join(['../models/'+self.name, str(self.input_shape[0]), self.run_id,  'col_names.txt'])
            self._save_cols(col_name)
            
    def _save_cols(self, save_adr):
        with open(save_adr, 'w') as fw:
            for col in self.cols:
                fw.write(col +'\n')
        

    def train(self, sub_sample):

        self._load_data(sub_sample)
        self._train_test_split()
        self._init_model(verbose=True)

        print('-' * 50)
        print('Training...')
        print('-' * 50)

        tb = TensorBoard(log_dir='./logs')
        best_weights_filepath = '../models/' + self.name + \
            '-weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
        checkpoint = ModelCheckpoint(best_weights_filepath, monitor='val_loss',
                                     verbose=self.verbose, save_best_only=True, mode='auto')  # mode max? val_loss val_acc

        print('Fitting  model...')
        hist_model = self.model.fit(self.X_train, self.y_train,
                                    batch_size=self.batch_size,
                                    epochs=self.nb_epoch,
                                    validation_data=(
                                        self.X_test, self.y_test),
                                    shuffle=True,
                                    verbose=self.verbose,
                                    callbacks=[tb, checkpoint])

        score = self.model.evaluate(
            self.X_test, self.y_test, verbose=self.verbose)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

    def train_all(self, nb_epoch):
        '''train using all data avaliable
        '''
        self._load_data()
        self._init_model(verbose=True)        
        print('Fitting  model...')

        self.model.fit(self.X_train, self.y_train,
                                    batch_size=self.batch_size,
                                    nb_epoch=nb_epoch,
                                    shuffle=True,
                                    verbose=self.verbose)
        best_weights_filepath = '../models/{}.hdf5'.format(self.name)
        self.model.save_weights(best_weights_filepath, overwrite=True)
        score = self.model.evaluate(
            self.X_train, self.y_train, verbose=self.verbose)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

    def cross_validation(self):
        ''' do a 6 fold cross-validation, draw ROC curve
        '''

        self._load_data()

        mean_tpr = 0.0
        mean_fpr = numpy.linspace(0, 1, 100)
        colors = ['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange']
        lw = 2
        i = 0

        pdf = PdfPages('../data/cnn_cv.pdf')

        cvscores = []
        kfold = StratifiedKFold(n_splits=6, shuffle=True, random_state=seed)
        for (train, test), color in zip(kfold.split(self.X_train, self.y_train), colors):
            
            self._init_model(verbose=False)
            # Fit the model
            self.model.fit(self.X_train[train], self.y_train[train],
                           nb_epoch=self.nb_epoch, batch_size=self.batch_size,
                           verbose=self.verbose)
            # evaluate the model
            scores = self.model.evaluate(
                self.X_train[test], self.y_train[test], verbose=self.verbose)
            print("%s: %.2f%%" %
                  (self.model.metrics_names[1], scores[1] * 100))
            cvscores.append(scores[1] * 100)

            # Compute ROC curve and area the curve, mean ROC using interpolation
            probas_ = self.model.predict(self.X_train[test])
            fpr, tpr, thresholds = roc_curve(self.y_train[test], probas_[:, 0])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=lw, color=color,
                     label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
            i += 1

        cv_results = "%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores))

        plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Luck')
        mean_tpr /= kfold.get_n_splits(self.X_train, self.y_train)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
                 label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Training cross-validation ROC\nAccuracy:' + cv_results)
        plt.legend(loc="lower right")
        plt.show()
        pdf.savefig(bbox_inches='tight')    
        pdf.close()
        plt.close() 


    def pred(self, get_last_layer, layer_index=-3):

        # get predicted probability
        if self.train_flag:
            self.data['training'] = 1
            self.data.loc[self.idx_test, 'training'] = 0
            proba = self.model.predict(self.X_pred, batch_size=32)
        else:
            self._load_data()
            self._init_model(verbose=False)
            proba = self.model.predict(self.X_pred, batch_size=32)

        # get intermidate layer value
        if get_last_layer:
            self.get_ith_layer_output(layer_index)

        # write output
        self.data['cnn_prob'] = proba
        self.data.to_csv(self.f_out)

    def get_ith_layer_output(self, i=-1, mode='test'):
        ''' see https://keras.io/getting-started/faq/#keras-faq-frequently-asked-keras-questions'''
        get_ith_layer = K.function(
            [self.model.layers[0].input, K.learning_phase()], [self.model.layers[i].output])
        layer_output = get_ith_layer(
            [self.X_pred, 0 if mode == 'test' else 1])[0]
        num_neuro = layer_output.shape[1]
        columns = []
        for i in range(num_neuro):
            columns.append('neuron_' + str(i))
        df = pd.DataFrame(layer_output, columns=columns)
        self.data = pd.concat([self.data, df], axis=1)


class CNN_Model_Mode6(CNN_Model):
    def __init__(self, input_shape=(20, 1, 1), weights_path=None, name='resi_model_mode1',
                 train_flag=True, nb_epoch=50, batch_size=64, verbose=0,
                 fname='../data/input_data.csv', f_out='../data/output_data.csv'):
        super(CNN_Model_Mode6, self).__init__(input_shape=input_shape, weights_path=weights_path, name=name,
                                              train_flag=train_flag, nb_epoch=nb_epoch, batch_size=batch_size, verbose=verbose,
                                              fname=fname, f_out=f_out)

    def _init_model(self, verbose):
                # number of convolutional filters to use
        nb_filters = 32
        # size of pooling area for max pooling
        pool_size = (2, 1)
        # convolution kernel size
        kernel_size = (3, 1)


        input_ = Input(shape=self.input_shape, name='input')
        x = Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode="same", activation="relu")(input_)
        for _ in range(2):
            y = Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode="same", activation="relu")(x)
            y = Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode="same")(y)
            x = merge([x, y], mode="sum")
            x = Activation("relu")(x)
            x = MaxPooling2D(pool_size=pool_size)(x)

        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        x = Dense(1, name='dense1')(x)
        act2 = Activation('sigmoid', name='act2')(x)

        self.model = Model(input=input_, output=act2)
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

        if self.weights_path:
            self.model.load_weights(self.weights_path)

        if verbose:
            # model summary and save arch
            print self.model.summary()

            outname = '../models/'+self.name + '_' + str(self.input_shape[0]) + 'cols'+ '_' + self.run_id +'.png'
            plot_model(self.model, show_shapes=True, to_file=outname)
            col_name = '../models/'+self.name + '_' + str(self.input_shape[0]) + 'cols'+ '_' + self.run_id +'.txt'
            self._save_cols(col_name)

