import tensorflow as tf
import dataset
import argparse
import pandas as pd
import numpy as np

def predict(model_path, gene_type, input_path, output_path):
    model = tf.keras.models.load_model(model_path)
    model.summary()

    x, var_id = dataset.get_test(input_path, gene_type)
    pred = model.predict(x)
    pred = np.squeeze(pred, axis=1)

    df = pd.DataFrame({'var_id': var_id, 'pred': pred})
    df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--gene_type', type=str, required=True)
    args = parser.parse_args()
    predict(args.model_path, args.gene_type, args.input_path, args.output_path)


if __name__ == '__main__':
    main()
