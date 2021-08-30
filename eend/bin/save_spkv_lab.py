#!/usr/bin/env python3

# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved

import yamlargparse
from eend.pytorch_backend.infer import save_spkv_lab

parser = yamlargparse.ArgumentParser(description='decoding')
parser.add_argument('-c', '--config', help='config file path',
                    action=yamlargparse.ActionConfigFile)
parser.add_argument('data_dir',
                    help='kaldi-style data dir')
parser.add_argument('model_file',
                    help='best.nnet')
parser.add_argument('out_dir',
                    help='output directory.')

# The following arguments are set in conf/save_spkv_lab.yaml
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

args = parser.parse_args()
print(args)

save_spkv_lab(args)
