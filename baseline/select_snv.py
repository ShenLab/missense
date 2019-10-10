import argparse
import glob
import random

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input_prefix', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--ratio', type=float, default=2e-3)
    args=parser.parse_args()

    #inputs=glob.glob('{}'.format(args.input_prefix))
    inputs=[args.input_prefix]
    random.seed(2018)

    with open(args.output_path, 'w') as fw:
        for idx, input_path in enumerate(inputs):
            print input_path
            with open(input_path) as f:
                cnt=0
                for line in f:
                    if idx == 0 and  cnt == 0:
                        fw.write(line)
                    if cnt > 0 and random.random() < args.ratio: 
                        fw.write(line)
                    cnt += 1

