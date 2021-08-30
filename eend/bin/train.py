#!/usr/bin/env python3

# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved

import os
import yamlargparse
from eend.pytorch_backend.train import train, save_feature

parser = yamlargparse.ArgumentParser(description='training')
parser.add_argument('-c', '--config', help='config file path',
                    action=yamlargparse.ActionConfigFile)
parser.add_argument('train_data_dir',
                    help='kaldi-style data dir used for training.')
parser.add_argument('valid_data_dir',
                    help='kaldi-style data dir used for validation.')
parser.add_argument('model_save_dir',
                    help='output directory which model file will be saved in.')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--spkv-lab', default='',
                    help='file path of speaker vector with label and\
                    speaker ID conversion table for adaptation')

# The following arguments are set in conf/train.yaml or conf/adapt.yaml
parser.add_argument('--spk-loss-ratio', default=0.03, type=float)
parser.add_argument('--spkv-dim', default=256, type=int,
                    help='dimension of speaker embedding vector')
parser.add_argument('--max-epochs', default=100, type=int,
                    help='Max. number of epochs to train')
parser.add_argument('--input-transform', default='logmel23_mn',
                    choices=['', 'log', 'logmel', 'logmel23', 'logmel23_mn',
                             'logmel23_mvn', 'logmel23_swn'],
                    help='input transform')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--optimizer', default='noam', type=str)
parser.add_argument('--num-speakers', default=3, type=int)
parser.add_argument('--gradclip', default=5, type=int,
                    help='gradient clipping. if < 0, no clipping')
parser.add_argument('--chunk-size', default=150, type=int,
                    help='number of frames in one utterance')
parser.add_argument('--batchsize', default=64, type=int,
                    help='number of utterances in one batch.\
                    Note that real batchsize = number of gpu *\
                    batchsize-per-gpu * batchsize')
parser.add_argument('--num-workers', default=8, type=int)
parser.add_argument('--hidden-size', default=256, type=int)
parser.add_argument('--context-size', default=7, type=int)
parser.add_argument('--subsampling', default=10, type=int)
parser.add_argument('--frame-size', default=200, type=int)
parser.add_argument('--frame-shift', default=80, type=int)
parser.add_argument('--sampling-rate', default=8000, type=int)
parser.add_argument('--noam-scale', default=1.0, type=float)
parser.add_argument('--noam-warmup-steps', default=25000, type=float)
parser.add_argument('--transformer-encoder-n-heads', default=8, type=int)
parser.add_argument('--transformer-encoder-n-layers', default=6, type=int)
parser.add_argument('--transformer-encoder-dropout', default=0.1, type=float)
parser.add_argument('--seed', default=777, type=int)
parser.add_argument('--feature-nj', default=100, type=int,
                    help='maximum number of subdirectories to store\
                    featlab_XXXXXXXX.npy')
parser.add_argument('--batchsize-per-gpu', default=16, type=int,
                    help='virtual_minibatch_size in padertorch')
parser.add_argument('--test-run', default=0, type=int, choices=[0, 1],
                    help='padertorch test run switch; 1 is on, 0 is off')

args = parser.parse_args()
print(args)

# To speed up the training process, we first calculate input features
# to NN and save shuffled feature data to the disc. During training,
# we simply read the saved data from the disc.
path = '{}/data/.done'.format(args.model_save_dir)
is_file = os.path.isfile(path)
if is_file:
    print("skip feature saving.")
    train(args)
else:
    save_feature(args)
    train(args)
