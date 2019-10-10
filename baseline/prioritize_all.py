import model
import pandas
import argparse
import json
import input_data
import glob
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_config', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--input_prefix', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    args = parser.parse_args()

    with open(args.input_config) as f:
        input_config = json.load(f)
        print input_config
    dataset = input_data.Dataset(input_config)

    model_name = args.model_name

    m = model.Model(dataset, model_name)
    m.load_model(args.model_dir, input_config['cv'])
    inputs = glob.glob('{}*.csv'.format(args.input_prefix))
    for input_path in inputs:
        dataset.reload('all_snv', input_path)
        preds, _ = m.predict('all_snv')
        pred = np.mean(preds, axis=0)
        print pred.shape
        output = '{}/{}.{}'.format(args.output_dir, model_name,
                                   input_path.split('/')[-1])
        np.savetxt(output, pred, fmt='%.8f')
