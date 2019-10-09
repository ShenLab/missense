import model
import input_data
import argparse
import sys,os
import json
import model
import numpy as np
import random

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, type=str, choices=['train', 'test'])
    parser.add_argument('--input_config', type=str, required=True)
    parser.add_argument('--job_dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--pred_dir', type=str)
    args=parser.parse_args()

    seed=2018
    np.random.seed(seed)
    random.seed(seed)

    with open(args.input_config) as f:
        input_config=json.load(f)
        print input_config
    dataset=input_data.Dataset(input_config)

    rf=model.ModelRF(dataset)
    if args.mode == 'train':
        rf.train()
        rf.evaluate_auc("cancer")
        rf.evaluate_denovo()
        if args.model_dir is None:
            args.model_dir = '{}/model'.format(args.job_dir)
            if not os.path.exists(args.model_dir):
                os.makedirs(args.model_dir)
        rf.save_model(args.model_dir)

    if args.mode == 'test':
        if args.pred_dir is None:
            args.pred_dir = '{}/pred'.format(args.job_dir)
            os.makedirs(args.pred_dir)
        if args.model_dir is None:
            args.model_dir = '{}/model'.format(args.job_dir)
        rf.load_model(args.model_dir, input_config['cv'])
        def predict_one(name):
            preds, target = rf.predict(name)
            pred=np.mean(preds, axis=0)
            res=dataset.test[name]
            res['RF']=pred
            res.to_csv('{}/{}_{}.csv'.format(args.pred_dir, name, input_config['gene_type']),
                sep=',', index=False)
        predict_one('cancer')
        predict_one('asd')
        predict_one('chd')
        predict_one('control')

