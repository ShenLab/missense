
from models import CNN_Model, CNN_Model_Mode6


weights_path = None #'../models/cnn_model.hdf5'
model = CNN_Model_Mode6(weights_path=weights_path, train_flag=True, verbose=1,
                       nb_epoch=10, batch_size=64, 
                       fname='../data/input_data.csv', f_out = '../data/output/output_data_mode5.csv')

model.train(sub_sample=True)
model.pred(get_last_layer=False)



