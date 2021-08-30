# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved

import os
import numpy as np
from functools import partial
import torch
from eend.feature import get_input_dim
from eend.pytorch_backend.models import fix_state_dict
from eend.pytorch_backend.models import PadertorchModel
from eend.pytorch_backend.models import TransformerDiarization
from eend.pytorch_backend.transformer import NoamScheduler
from eend.pytorch_backend.diarization_dataset \
    import DiarizationDatasetFromWave, DiarizationDatasetFromFeat
import padertorch as pt
import padertorch.train.optimizer as pt_opt


def collate_fn_ns(batch, n_speakers, spkidx_tbl):
    xs, ts, ss, ns, ilens = list(zip(*batch))
    valid_chunk_indices1 = [i for i in range(len(ts))
                            if ts[i].shape[1] == n_speakers]
    valid_chunk_indices2 = []

    # n_speakers (rec-data) > n_speakers (model)
    invalid_chunk_indices1 = [i for i in range(len(ts))
                              if ts[i].shape[1] > n_speakers]

    ts = list(ts)
    ss = list(ss)
    for i in invalid_chunk_indices1:
        s = np.sum(ts[i], axis=0)
        cs = ts[i].shape[0]
        if len(s[s > 0.5]) <= n_speakers:
            # n_speakers (chunk-data) <= n_speakers (model)
            # update valid_chunk_indices2
            valid_chunk_indices2.append(i)
            idx_arr = np.where(s > 0.5)[0]
            ts[i] = ts[i][:, idx_arr]
            ss[i] = ss[i][idx_arr]
            if len(s[s > 0.5]) < n_speakers:
                # n_speakers (chunk-data) < n_speakers (model)
                # update ts[i] and ss[i]
                n_speakers_real = len(s[s > 0.5])
                zeros_ts = np.zeros((cs, n_speakers), dtype=np.float32)
                zeros_ts[:, :-(n_speakers-n_speakers_real)] = ts[i]
                ts[i] = zeros_ts
                mones_ss = -1 * np.ones((n_speakers,), dtype=np.int64)
                mones_ss[:-(n_speakers-n_speakers_real)] = ss[i]
                ss[i] = mones_ss
            else:
                # n_speakers (chunk-data) == n_speakers (model)
                pass
        else:
            # n_speakers (chunk-data) > n_speakers (model)
            pass

    # valid_chunk_indices: chunk indices using for training
    valid_chunk_indices = sorted(valid_chunk_indices1 + valid_chunk_indices2)

    ilens = np.array(ilens)
    ilens = ilens[valid_chunk_indices]
    ns = np.array(ns)[valid_chunk_indices]
    ss = np.array([ss[i] for i in range(len(ss))
                  if ts[i].shape[1] == n_speakers])
    xs = [xs[i] for i in range(len(xs)) if ts[i].shape[1] == n_speakers]
    ts = [ts[i] for i in range(len(ts)) if ts[i].shape[1] == n_speakers]
    xs = np.array([np.pad(x, [(0, np.max(ilens) - len(x)), (0, 0)],
                          'constant', constant_values=(-1,)) for x in xs])
    ts = np.array([np.pad(t, [(0, np.max(ilens) - len(t)), (0, 0)],
                          'constant', constant_values=(+1,)) for t in ts])

    if spkidx_tbl is not None:
        # Update global speaker ID
        all_n_speakers = np.max(spkidx_tbl) + 1
        bs = len(ns)
        ns = np.array([
                np.arange(
                    all_n_speakers,
                    dtype=np.int64
                    ).reshape(all_n_speakers, 1)] * bs)
        ss = np.array([spkidx_tbl[ss[i]] for i in range(len(ss))])

    return (xs, ts, ss, ns, ilens)


def collate_fn(batch):
    xs, ts, ss, ns, ilens = list(zip(*batch))
    ilens = np.array(ilens)
    xs = np.array([np.pad(
        x, [(0, np.max(ilens) - len(x)), (0, 0)],
        'constant', constant_values=(-1,)
        ) for x in xs])
    ts = np.array([np.pad(
        t, [(0, np.max(ilens) - len(t)), (0, 0)],
        'constant', constant_values=(+1,)
        ) for t in ts])
    ss = np.array(ss)
    ns = np.array(ns)

    return (xs, ts, ss, ns, ilens)


def train(args):
    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTORCH_SEED'] = str(args.seed)
    torch.backends.cudnn.benchmark = False

    # Prepare data
    featlab_chunk_indices_path =\
        '{}/data/featlab_chunk_indices.txt'.format(args.model_save_dir)

    featdim = get_input_dim(args.frame_size,
                            args.context_size,
                            args.input_transform)

    train_set = DiarizationDatasetFromFeat(
        featlab_chunk_indices_path,
        featdim,
        )
    dev_set = DiarizationDatasetFromWave(
        args.valid_data_dir,
        chunk_size=args.chunk_size,
        context_size=args.context_size,
        input_transform=args.input_transform,
        frame_size=args.frame_size,
        frame_shift=args.frame_shift,
        subsampling=args.subsampling,
        rate=args.sampling_rate,
        n_speakers=args.num_speakers,
        )

    # Prepare model
    all_n_speakers = train_set.get_allnspk()
    net = TransformerDiarization(
            args.num_speakers,
            featdim,
            n_units=args.hidden_size,
            n_heads=args.transformer_encoder_n_heads,
            n_layers=args.transformer_encoder_n_layers,
            dropout_rate=args.transformer_encoder_dropout,
            all_n_speakers=all_n_speakers,
            d=args.spkv_dim)

    if args.initmodel:
        # adaptation
        model_parameter_dict = torch.load(args.initmodel)['model']
        fix_model_parameter_dict = fix_state_dict(model_parameter_dict)
        all_n_speakers = fix_model_parameter_dict["embed.weight"].shape[0]

        print("old all_n_speakers : {}".format(all_n_speakers))
        net = TransformerDiarization(
                args.num_speakers,
                featdim,
                n_units=args.hidden_size,
                n_heads=args.transformer_encoder_n_heads,
                n_layers=args.transformer_encoder_n_layers,
                dropout_rate=args.transformer_encoder_dropout,
                all_n_speakers=all_n_speakers,
                d=args.spkv_dim)
        net.load_state_dict(fix_model_parameter_dict)
        npz = np.load(args.spkv_lab)
        spkvecs = npz['arr_0']
        spklabs = npz['arr_1']
        spkidx_tbl = npz['arr_2']

        # init
        spk_num = len(np.unique(spklabs))
        fet_dim = spkvecs.shape[1]
        fet_arr = np.zeros([spk_num, fet_dim])

        # sum
        bs = spklabs.shape[0]
        for i in range(bs):
            if spkidx_tbl[spklabs[i]] == -1:
                raise ValueError(spklabs[i])
            fet_arr[spkidx_tbl[spklabs[i]]] += spkvecs[i]

        # normalize
        for spk in range(spk_num):
            org = fet_arr[spk]
            norm = np.linalg.norm(org, ord=2)
            fet_arr[spk] = org / norm

        weight = torch.from_numpy(fet_arr.astype(np.float32)).clone()
        print("new all_n_speakers : {}".format(weight.shape[0]))

        print(net)
        net.modfy_emb(weight)
        print(net)

    device = [device_id for device_id in range(torch.cuda.device_count())]
    model = PadertorchModel(net=net)
    print('GPU device {} is used'.format(device))
    print('Prepared model.')

    # Setup optimizer
    if args.optimizer == 'adam':
        optimizer = pt_opt.Adam(lr=args.lr, gradient_clipping=args.gradclip)
    elif args.optimizer == 'sgd':
        optimizer = pt_opt.SGD(lr=args.lr, gradient_clipping=args.gradclip)
    elif args.optimizer == 'noam':
        optimizer = pt_opt.Adam(lr=args.lr, betas=(0.9, 0.98), eps=1e-9,
                                gradient_clipping=args.gradclip)
    else:
        raise ValueError(args.optimizer)

    trainloader = torch.utils.data.DataLoader(train_set,
                                              batch_size=args.batchsize,
                                              shuffle=False,
                                              num_workers=args.num_workers)

    # pit_loss_ratio means diarization loss ratio
    pit_loss_ratio = abs(1 - args.spk_loss_ratio)
    spk_loss_ratio = args.spk_loss_ratio
    virtual_minibatch_size = len(device) * args.batchsize_per_gpu

    trainer = pt.trainer.Trainer(
            model,
            args.model_save_dir,
            optimizer,
            stop_trigger=(args.max_epochs, 'epoch'),
            summary_trigger=(1, 'iteration'),
            virtual_minibatch_size=virtual_minibatch_size,
            loss_weights={
              "pit_loss": pit_loss_ratio,
              "spk_loss": spk_loss_ratio,
              }
            )

    devloader = torch.utils.data.DataLoader(dev_set, batch_size=args.batchsize,
                                            shuffle=False,
                                            num_workers=args.num_workers,
                                            collate_fn=collate_fn)
    if args.test_run == 1:
        trainer.test_run(trainloader, devloader)

    trainer.register_validation_hook(validation_iterator=devloader,
                                     max_checkpoints=args.max_epochs+1)

    # learning rate scheduler
    if args.optimizer == 'noam':
        scheduler = NoamScheduler(trainer.optimizer.optimizer,
                                  args.hidden_size,
                                  warmup_steps=args.noam_warmup_steps,
                                  tot_step=len(trainloader),
                                  scale=1.0)
        trainer.register_hook(
            pt.train.hooks.LRSchedulerHook(scheduler, trigger=(1, 'iteration'))
            )

    trainer.train(trainloader, resume=False, device=device)
    print('Finished!')


def save_feature(args):
    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTORCH_SEED'] = str(args.seed)
    torch.backends.cudnn.benchmark = False

    device = [device_id for device_id in range(torch.cuda.device_count())]
    print('GPU device {} is used'.format(device))

    train_set = DiarizationDatasetFromWave(
        args.train_data_dir,
        chunk_size=args.chunk_size,
        context_size=args.context_size,
        input_transform=args.input_transform,
        frame_size=args.frame_size,
        frame_shift=args.frame_shift,
        subsampling=args.subsampling,
        rate=args.sampling_rate,
        n_speakers=args.num_speakers,
        )

    # Count n_chunks
    batchsize = args.batchsize * len(device) * \
        args.batchsize_per_gpu
    f = open('{}/batchsize.txt'.format(args.model_save_dir), 'w')
    f.write("{}\n".format(batchsize))
    f.close()
    trainloader = torch.utils.data.DataLoader(
            train_set, batch_size=batchsize,
            shuffle=False, num_workers=args.num_workers,
            collate_fn=partial(
                collate_fn_ns,
                n_speakers=args.num_speakers,
                spkidx_tbl=None)
            )
    n_chunks = len(trainloader)
    print("n_chunks : {}".format(n_chunks))
    os.makedirs("{}/data/".format(args.model_save_dir), exist_ok=True)
    f = open('{}/data/n_chunks.txt'.format(args.model_save_dir), 'w')
    f.write("{}\n".format(n_chunks))
    f.close()

    if n_chunks % args.feature_nj == 0:
        max_num_per_dir = n_chunks // args.feature_nj
    else:
        max_num_per_dir = n_chunks // args.feature_nj + 1
    print("max_num_per_dir : {}".format(max_num_per_dir))

    # Save featlab_XXXXXXXX.npy and featlab_chunk_indices.txt
    spkidx_tbl = None
    if args.initmodel:
        # adaptation
        npz = np.load(args.spkv_lab)
        spkidx_tbl = npz['arr_2']

    torch.manual_seed(args.seed)
    trainloader = torch.utils.data.DataLoader(
            train_set, batch_size=batchsize,
            shuffle=True, num_workers=args.num_workers,
            collate_fn=partial(
                collate_fn_ns,
                n_speakers=args.num_speakers,
                spkidx_tbl=spkidx_tbl)
            )
    f = open('{}/data/featlab_chunk_indices.txt'.
             format(args.model_save_dir), 'w')
    idx = 0
    digit_num = len(str(args.feature_nj-1))
    fmt = "{}/data/{:0={}}/featlab_{:0=8}.npy"
    for data in trainloader:
        dir_num = idx // max_num_per_dir
        os.makedirs("{}/data/{:0={}}/".
                    format(args.model_save_dir, dir_num, digit_num),
                    exist_ok=True)
        output_npy_path = fmt.format(args.model_save_dir,
                                     dir_num, digit_num, idx)
        print(output_npy_path)
        bs = data[0].shape[0]
        cs = data[0].shape[1]
        # data0 (feature)
        data0 = data[0]
        # data1 (reference speech activity)
        data1 = data[1]
        # data2 (reference speaker ID)
        data2 = np.zeros([bs, cs, data[2].shape[1]], dtype=np.float32)
        for j in range(bs):
            data2[j, :, :] = data[2][j, :]
        # data3 (reference number of all speakers)
        data3 = np.ones([bs, cs, 1], dtype=np.float32) * len(data[3][0])
        # data4 (real chunk size)
        data4 = np.zeros([bs, cs, 1], dtype=np.float32)
        for j in range(bs):
            data4[j, :, :] = data[4][j]
        save_data = np.concatenate((data0,
                                    data1,
                                    data2,
                                    data3,
                                    data4), axis=2)

        np.save(output_npy_path, save_data)
        for j in range(save_data.shape[0]):
            f.write("{} {}\n".format(output_npy_path, j))
        idx += 1
    f.close()

    # Create completion flag
    f = open('{}/data/.done'.format(args.model_save_dir), 'w')
    f.write("")
    f.close()
    print('Finished!')
