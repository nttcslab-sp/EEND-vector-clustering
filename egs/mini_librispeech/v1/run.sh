#!/bin/bash

# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved

stage=0

# Base config files for {train,infer}.py
train_config=conf/train.yaml
infer0_config=conf/infer_est_nspk0.yaml
infer1_config=conf/infer_est_nspk1.yaml

# Additional arguments passed to {train,infer}.py
# You need not edit the base config files above
train_args=
infer_args=

# Model averaging options
average_start=8
average_end=10

. path.sh
. cmd.sh
. parse_options.sh || exit

train_set=data/simu/data/train_clean_5_ns3_beta2_500
valid_set=data/simu/data/dev_clean_2_ns3_beta2_500

if [ $stage -le 0 ]; then
    echo -e "==== stage 0: prepare data ===="
    ./run_prepare_shared.sh --simu_opts_num_speaker 3
    # Note that for simplicity we generate data/simu/data/*/utt2spk by using local/data_prep.sh,
    # then speaker ID is regarded as [reader]-[chapter]
fi

set -eu
# Parse the config file to set bash variables like: $infer0_frame_shift, $infer1_subsampling
eval `yaml2bash.py --prefix infer0 $infer0_config`
eval `yaml2bash.py --prefix infer1 $infer1_config`

# Append gpu reservation flag to the queuing command
train_cmd+=" --gpu 1"
infer_cmd+=" --gpu 1"

# Build directry names for an experiment
#  - Training
#     exp/diarize/model/${train_id}.${valid_id}.${train_config_id}
#  - Inference
#     exp/diarize/infer/${train_id}.${valid_id}.${train_config_id}.${infer0_config_id}
#     exp/diarize/infer/${train_id}.${valid_id}.${train_config_id}.${infer1_config_id}
#  - Scoring
#     exp/diarize/scoring/${train_id}.${valid_id}.${train_config_id}.${infer0_config_id}
#     exp/diarize/scoring/${train_id}.${valid_id}.${train_config_id}.${infer1_config_id}
train_id=$(basename $train_set)
valid_id=$(basename $valid_set)
train_config_id=$(echo $train_config | sed -e 's%conf/%%' -e 's%/%_%' -e 's%\.yaml$%%')
infer0_config_id=$(echo $infer0_config | sed -e 's%conf/%%' -e 's%/%_%' -e 's%\.yaml$%%')
infer1_config_id=$(echo $infer1_config | sed -e 's%conf/%%' -e 's%/%_%' -e 's%\.yaml$%%')

# Additional arguments are added to config_id
train_config_id+=$(echo $train_args | sed -e 's/\-\-/_/g' -e 's/=//g' -e 's/ \+//g')
infer0_config_id+=$(echo $infer_args | sed -e 's/\-\-/_/g' -e 's/=//g' -e 's/ \+//g')
infer1_config_id+=$(echo $infer_args | sed -e 's/\-\-/_/g' -e 's/=//g' -e 's/ \+//g')

model_id=$train_id.$valid_id.$train_config_id
model_dir=exp/diarize/model/$model_id

if [ $stage -le 1 ]; then
    echo -e "\n==== stage 1: training model with simulated mixtures ===="
    # To speed up the training process, we first calculate input features
    # to NN and save shuffled feature data to the disc. During training,
    # we simply read the saved data from the disc.
    echo "training model at $model_dir"
    if [ -d $model_dir/checkpoints ]; then
        echo "$model_dir/checkpoints already exists. "
        echo " if you want to retry, please remove it."
        exit 1
    fi
    work=$model_dir/.work
    mkdir -p $work
    $train_cmd $work/train.log \
        train.py \
            -c $train_config \
            $train_args \
            $train_set $valid_set $model_dir \
            || exit 1
fi

ave_id=avg${average_start}-${average_end}
if [ $stage -le 2 ]; then
    echo -e "\n==== stage 2: averaging trained models ===="
    echo "averaging model parameters into $model_dir/checkpoints/$ave_id.nnet.pth"
    if [ -s $model_dir/checkpoints/$ave_id.nnet.pth ]; then
        echo "$model_dir/checkpoints/$ave_id.nnet.pth already exists. "
    fi
    last_epoch=$(ls $model_dir/checkpoints/ckpt_[0-9]*.pth | grep -v "/ckpt_0.pth"$ | wc -l)
    echo -e "last epoch of existence : $last_epoch"
    if [ $last_epoch -lt $average_end ]; then
        echo -e "error : average_end $average_end is too large."
        exit 1
    fi
    models=$(ls $model_dir/checkpoints/ckpt_[0-9]*.pth -tr | head -n $((${average_end}+1)) | tail -n $((${average_end}-${average_start}+1)))
    echo -e "take the average with the following models:"
    echo -e $models | tr " " "\n"
    model_averaging.py $model_dir/checkpoints/$ave_id.nnet.pth $models || exit 1
fi

infer_dir=exp/diarize/infer/$model_id.$ave_id.$infer0_config_id
if [ $stage -le 3 ]; then
    echo -e "\n==== stage 3: inference for evaluation (speaker counting: oracle) ===="
    echo "inference at $infer_dir"
    if [ -d $infer_dir ]; then
        echo "$infer_dir already exists. "
    fi
    for dset in dev_clean_2_ns3_beta2_500; do
        work=$infer_dir/$dset/.work
        mkdir -p $work
        $infer_cmd $work/infer.log \
            infer.py -c $infer0_config \
            $infer_args \
            data/simu/data/${dset} \
            $model_dir/checkpoints/$ave_id.nnet.pth \
            $infer_dir/$dset \
            || exit 1
    done
fi

scoring_dir=exp/diarize/scoring/$model_id.$ave_id.$infer0_config_id
if [ $stage -le 4 ]; then
    echo -e "\n==== stage 4: scoring for evaluation (speaker counting: oracle) ===="
    echo "scoring at $scoring_dir"
    if [ -d $scoring_dir ]; then
        echo "$scoring_dir already exists. "
    fi
    for dset in dev_clean_2_ns3_beta2_500; do
        work=$scoring_dir/$dset/.work
        mkdir -p $work
        find $infer_dir/$dset -iname "*.h5" > $work/file_list_$dset
        for med in 1 11; do
        for th in 0.3 0.4 0.5 0.6 0.7; do
            make_rttm.py --median=$med --threshold=$th \
                --frame_shift=$infer0_frame_shift --subsampling=$infer0_subsampling --sampling_rate=$infer0_sampling_rate \
                $work/file_list_$dset $scoring_dir/$dset/hyp_${th}_$med.rttm
            md-eval.pl -c 0.25 \
                -r data/simu/data/$dset/rttm \
                -s $scoring_dir/$dset/hyp_${th}_$med.rttm > $scoring_dir/$dset/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
        done
        done
        best_score.sh $scoring_dir/$dset
    done
fi

infer_dir=exp/diarize/infer/$model_id.$ave_id.$infer1_config_id
if [ $stage -le 5 ]; then
    echo -e "\n==== stage 5: inference for evaluation (speaker counting: estimated) ===="
    echo "inference at $infer_dir"
    if [ -d $infer_dir ]; then
        echo "$infer_dir already exists. "
    fi
    for dset in dev_clean_2_ns3_beta2_500; do
        work=$infer_dir/$dset/.work
        mkdir -p $work
        $infer_cmd $work/infer.log \
            infer.py -c $infer1_config \
            $infer_args \
            data/simu/data/${dset} \
            $model_dir/checkpoints/$ave_id.nnet.pth \
            $infer_dir/$dset \
            || exit 1
    done
fi

scoring_dir=exp/diarize/scoring/$model_id.$ave_id.$infer1_config_id
if [ $stage -le 6 ]; then
    echo -e "\n==== stage 6: scoring for evaluation (speaker counting: estimated) ===="
    echo "scoring at $scoring_dir"
    if [ -d $scoring_dir ]; then
        echo "$scoring_dir already exists. "
    fi
    for dset in dev_clean_2_ns3_beta2_500; do
        work=$scoring_dir/$dset/.work
        mkdir -p $work
        find $infer_dir/$dset -iname "*.h5" > $work/file_list_$dset
        for med in 1 11; do
        for th in 0.3 0.4 0.5 0.6 0.7; do
            make_rttm.py --median=$med --threshold=$th \
                --frame_shift=$infer1_frame_shift --subsampling=$infer1_subsampling --sampling_rate=$infer1_sampling_rate \
                $work/file_list_$dset $scoring_dir/$dset/hyp_${th}_$med.rttm
            md-eval.pl -c 0.25 \
                -r data/simu/data/$dset/rttm \
                -s $scoring_dir/$dset/hyp_${th}_$med.rttm > $scoring_dir/$dset/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
        done
        done
        best_score.sh $scoring_dir/$dset
    done
fi

echo "Finished !"
