import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import datetime
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Lambda, merge
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils
from keras import backend as K
from IPython.display import SVG, display
from keras.utils.vis_utils import plot_model
import numpy

import pandas as pd

from random import shuffle
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import sys


seed = 2017
numpy.random.seed(seed)


class CNN_Model(object):

    def __init__(self, input_shape=(20, 1, 1), weights_path=None, name='resnet_model',
                 train_flag=True, nb_epoch=50, batch_size=64, verbose=0,
                 exclude_cols = {}, 
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
        self.exclude_cols = exclude_cols

    def _load_data(self, sub_sample=False):
        '''load data are not in exclude_cols into self.X_pred, subsample negative if specified
           load feathers to self.X_train
           load target in self.y is exist
        '''
        print('Loading training data...')
        self.data = pd.read_csv(self.fname)

        if sub_sample:
            pos = self.data[self.data['target'] == 1]
            neg = self.data[self.data['target'] == 0]
            # only sub sample negative
            if pos.shape[0] < neg.shape[0]:
                neg = neg.sample(min(neg.shape[0], pos.shape[0] * sub_sample))
            self.data = pd.concat([pos, neg], ignore_index=True)
            print 'pos:', pos.shape, 'neg:', neg.shape

        cols = [col for col in self.data.columns if col not in self.exclude_cols]

        # set attributes of all data used for prediction
        self.input_shape = (len(cols), 1, 1)
        self.X_pred = self.data[cols].values
        self.X_pred = self.X_pred.reshape(
            self.X_pred.shape[0], self.X_pred.shape[1], 1, 1)

        if 'target' in self.data.columns:
            self.y = self.data['target']
            self.X_train = self.X_pred
            self.y_train = self.y

    def _train_test_split(self):
        '''80/20 split, 80% for training, 20% for validation
        '''
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
        '''initial strucutre of resnet model
        '''

        # number of convolutional filters to use
        nb_filters = 32
        # size of pooling area for max pooling
        pool_size = (2, 1)
        # convolution kernel size
        kernel_size = (3, 1)

        activation = 'relu'

        # residual part
        input_ = Input(shape=self.input_shape, name='input')
        x = Conv2D(nb_filters, kernel_size, padding="same",
                   activation=activation)(input_)
        for _ in range(2):
            y = Conv2D(nb_filters, kernel_size, padding="same",
                       activation=activation)(x)
            y = Conv2D(nb_filters, kernel_size, padding="same")(y)
            x = merge([x, y], mode="sum")
            x = Activation(activation)(x)

        x = Flatten()(x)
        x = Dense(512, activation=activation)(x)
        x = Dense(1, name='dense1')(x)
        act2 = Activation('sigmoid', name='act2')(x)

        self.model = Model(input=input_, output=act2)
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

        if self.weights_path:
            self.model.load_weights(self.weights_path)

        # model summary
        if verbose:
            print self.model.summary()

    def train(self, sub_sample):

        self._load_data(sub_sample)
        self._train_test_split()
        self._init_model(verbose=True)

        print('-' * 50)
        print('Training...')
        print('-' * 50)

        # save best weight
        n_cols = self.X_train.shape[1]
        tb = TensorBoard(log_dir='./logs')
        best_weights_filepath = '../models/' + self.name + '-' + str(n_cols) + 'cols_' + datetime.datetime.now(
        ).strftime('%Y%m%d-%H.%M.%S') + '-weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
        checkpoint = ModelCheckpoint(best_weights_filepath, monitor='val_loss',
                                     verbose=self.verbose, save_best_only=True, mode='auto')

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

    def cross_validation(self, sub_sample=3):
        ''' do a 6 fold cross-validation and draw ROC curve
        '''

        self._load_data(sub_sample=sub_sample)

        # set up cross validation tpr/fpr
        mean_tpr = 0.0
        mean_fpr = numpy.linspace(0, 1, 100)
        colors = ['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange']
        lw = 1
        i = 1

        plt.figure(figsize=(10, 10))

        cvscores = []
        aucscores = []
        kfold = StratifiedKFold(n_splits=6, shuffle=True, random_state=seed)
        for (train, test), color in zip(kfold.split(self.X_train, self.y_train), colors):

            # Fit the model
            self._init_model(verbose=False)
            self.model.fit(self.X_train[train], self.y_train[train],
                           nb_epoch=self.nb_epoch,
                           batch_size=self.batch_size,
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

            aucscores.append(str(roc_auc))
            print 'roc_auc = ' + str(roc_auc)

            plt.plot(fpr, tpr, lw=lw, color=color,
                     label='ROC fold %d (area = %.2f)' % (i, roc_auc))
            i += 1

        cv_results = "%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores),
                                              numpy.std(cvscores))
        print 'cv result: ' + str(cv_results)

        plt.plot([0, 1], [0, 1], linestyle='--',
                 lw=lw, color='k', label='Luck')
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

    def pred(self):
        '''function used to get prediction from CNN models
        '''

        # get predicted probability
        self._load_data(sub_sample=False)
        self._init_model(verbose=False)
        proba = self.model.predict(self.X_pred, batch_size=32)

        # write prediction to dataframe and save to output
        self.data['cnn_prob'] = proba
        self.data.to_csv(self.f_out)