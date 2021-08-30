#!/usr/bin/env python3

# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved

import yamlargparse
from eend.pytorch_backend.infer import infer

parser = yamlargparse.ArgumentParser(description='decoding')
parser.add_argument('-c', '--config', help='config file path',
                    action=yamlargparse.ActionConfigFile)
parser.add_argument('data_dir',
                    help='kaldi-style data dir')
parser.add_argument('model_file',
                    help='best.nnet')
parser.add_argument('out_dir',
                    help='output directory.')

# The following arguments are set in conf/infer_est_nspk{0,1}.yaml
parser.add_argument('--est-nspk', default=1, type=int, choices=[0, 1],
                    help='At clustering stage, --est-nspk 0 means that\
                    oracle number of speakers is used, --est-nspk 1 means\
                    estimating numboer of speakers')
parser.add_argument('--num-speakers', default=3, type=int)
parser.add_argument('--spkv-dim', default=256, type=int,
                    help='dimension of speaker embedding vector')
parser.add_argument('--hidden-size', default=256, type=int)
parser.add_argument('--input-transform', default='logmel23_mn',
                    choices=['', 'log', 'logmel',
                             'logmel23', 'logmel23_swn', 'logmel23_mn'],
                    help='input transform')
parser.add_argument('--chunk-size', default=300, type=int,
                    help='input is chunked with this size')
parser.add_argument('--context-size', default=7, type=int,
                    help='frame splicing')
parser.add_argument('--subsampling', default=10, type=int)
parser.add_argument('--sampling-rate', default=8000, type=int,
                    help='sampling rate')
parser.add_argument('--frame-size', default=200, type=int,
                    help='frame size')
parser.add_argument('--frame-shift', default=80, type=int,
                    help='frame shift')
parser.add_argument('--transformer-encoder-n-heads', default=8, type=int)
parser.add_argument('--transformer-encoder-n-layers', default=6, type=int)
parser.add_argument('--sil-spk-th', default=0.05, type=float,
                    help='activity threshold to detect the silent speaker')
parser.add_argument('--ahc-dis-th', default=1.0, type=float,
                    help='distance threshold above which clusters\
                    will not be merged')
parser.add_argument('--clink-dis', default=1e+4, type=float,
                    help='modified distance corresponding to cannot-link')

args = parser.parse_args()
print(args)

infer(args)
