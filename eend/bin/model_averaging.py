#!/usr/bin/env python3

# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved

import torch
import argparse


def average_model_pytorch(ifiles, ofile):
    omodel = {}
    for path in ifiles:
        state_dict = torch.load(path)['model']
        for key in state_dict.keys():
            val = state_dict[key]
            if key not in omodel:
                omodel[key] = val
            else:
                omodel[key] += val
    for key in omodel.keys():
        omodel[key] /= len(ifiles)
    torch.save(dict(model=omodel), ofile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ofile")
    parser.add_argument("ifiles", nargs='+')
    args = parser.parse_args()
    average_model_pytorch(args.ifiles, args.ofile)
