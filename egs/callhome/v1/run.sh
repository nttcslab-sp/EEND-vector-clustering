#!/bin/bash

# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved

stage=0
db_path=/db # need to modify
simu_opts_num_speaker=3
simu_opts_sil_scale=10
simu_opts_num_train=100000
simu_opts_rvb_prob="0.1 --noise_snrs 10:15:20"

# Base config files for {train,save_spkv_lab,infer}.py
train_config=conf/train.yaml
save_spkv_lab_config=conf/save_spkv_lab.yaml
adapt_config=conf/adapt.yaml
infer0_config=conf/infer_est_nspk0.yaml
infer1_config=conf/infer_est_nspk1.yaml

# Additional arguments passed to {train,save_spkv_lab,infer}.py
# You need not edit the base config files above
train_args=
save_spkv_lab_args=
adapt_args=
infer_args=

# Model averaging options
average_start=91
average_end=100

# Adapted model averaging options
adapt_average_start=21
adapt_average_end=25

. path.sh
. cmd.sh
. parse_options.sh || exit

train_set=data/simu/data/swb_sre_tr_ns${simu_opts_num_speaker}_beta${simu_opts_sil_scale}_${simu_opts_num_train}
valid_set=data/simu/data/swb_sre_cv_ns${simu_opts_num_speaker}_beta${simu_opts_sil_scale}_500
adapt_set=data/eval/callhome1_spkall
adapt_valid_set=data/eval/callhome2_spk3
test_dsets=(callhome2_spk2 callhome2_spk3 callhome2_spk4 callhome2_spk5 callhome2_spk6 callhome2_spkall)

if [ $stage -le 0 ]; then
    echo -e "==== stage 0: prepare data ===="
    [ -L db ] && rm db
    ln -s $db_path
    [ ! -f musan.tar.gz ] && wget https://www.openslr.org/resources/17/musan.tar.gz
    [ ! -d musan ] && tar zvxf musan.tar.gz
    callhome_dir=$PWD/db/LDC2001S97
    swb2_phase1_train=$PWD/db/LDC98S75
    data_root=$PWD/db
    musan_root=$PWD/musan
    simu_actual_dirs=(\
    $PWD/export/c05/diarization-data
    $PWD/export/c08/diarization-data
    $PWD/export/c09/diarization-data)
    sad_opts="--extra-left-context 79 --extra-right-context 21 --frames-per-chunk 150 --extra-left-context-initial 0 --extra-right-context-final 0 --acwt 0.3 --cmd $sad_cmd"

    # Note: the following options are dummies
    # --simu_opts_num_speaker_array[1] "$simu_opts_num_speaker" \
    # --simu_opts_num_speaker_array[2] "$simu_opts_num_speaker" \
    # --simu_opts_num_speaker_array[3] "$simu_opts_num_speaker" \
    # --simu_opts_sil_scale_array[1] "$simu_opts_sil_scale" \
    # --simu_opts_sil_scale_array[2] "$simu_opts_sil_scale" \
    # --simu_opts_sil_scale_array[3] "$simu_opts_sil_scale" \

    ./run_prepare_shared_eda.sh \
        --callhome_dir "$callhome_dir" \
        --swb2_phase1_train "$swb2_phase1_train" \
        --data_root "$data_root" \
        --musan_root "$musan_root" \
        --simu_actual_dirs[0] "${simu_actual_dirs[0]}" \
        --simu_actual_dirs[1] "${simu_actual_dirs[1]}" \
        --simu_actual_dirs[2] "${simu_actual_dirs[2]}" \
        --sad_opts "$sad_opts" \
        --simu_opts_num_speaker_array[0] "$simu_opts_num_speaker" \
        --simu_opts_num_speaker_array[1] "$simu_opts_num_speaker" \
        --simu_opts_num_speaker_array[2] "$simu_opts_num_speaker" \
        --simu_opts_num_speaker_array[3] "$simu_opts_num_speaker" \
        --simu_opts_sil_scale_array[0] "$simu_opts_sil_scale" \
        --simu_opts_sil_scale_array[1] "$simu_opts_sil_scale" \
        --simu_opts_sil_scale_array[2] "$simu_opts_sil_scale" \
        --simu_opts_sil_scale_array[3] "$simu_opts_sil_scale" \
        --simu_opts_num_train "$simu_opts_num_train" \
        --simu_opts_rvb_prob "$simu_opts_rvb_prob"

    # Remove the following extra data
    rm -rf data/simu/data/swb_sre_cv_ns3n3n3n3_beta10n10n10n10_500
    rm -rf data/simu/data/swb_sre_tr_ns3n3n3n3_beta10n10n10n10_100000

    # Fix callhome1_spkall and callhome2_spkall
    for dset in callhome1_spkall callhome2_spkall; do
        # Remove the segment with zero duration from callhome1_spkall
        perl -p -i -e 's/iait_A_0035072_0035072\n//' data/eval/$dset/utt2spk
        perl -p -i -e 's/iait_A_0035072_0035072\n//' data/eval/$dset/segments
        # Modify speaker ID
        (cat data/eval/$dset/utt2spk \
             | grep "_0"$ \
             || perl -p -i -e 's/$/_0/' data/eval/$dset/utt2spk) > /dev/null
        # Update spk2utt and rttm
        LC_ALL=C utils/fix_data_dir.sh data/eval/$dset
        steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
            data/eval/$dset/utt2spk data/eval/$dset/segments \
            data/eval/$dset/rttm
    done

    # Separate callhome1_spkall and callhome2_spkall for each number of speakers
    for spk_num in spk2 spk3 spk4 spk5 spk6 spk7; do
    for dset in callhome1 callhome2; do
        [ "$spk_num" == "spk7" ] && [ "$dset" == "callhome2" ] && continue
        if ! validate_data_dir.sh --no-text --no-feats data/${dset}_${spk_num}; then
            copy_data_dir.sh data/${dset} data/${dset}_${spk_num}
            n_spk=$(echo $spk_num | perl -pe 's/spk//')
            echo "n_spk : $n_spk"
            # Extract ${n_spk}-speaker recordings in wav.scp
            utils/filter_scp.pl <(awk -v n_spk=${n_spk} '{if($2==n_spk) print;}'  data/${dset}/reco2num_spk) \
                data/${dset}/wav.scp > data/${dset}_${spk_num}/wav.scp
            # Regenerate segments file from fullref.rttm
            #  $2: recid, $4: start_time, $5: duration, $8: speakerid
            awk '{printf "%s_%s_%07d_%07d %s %.2f %.2f\n", \
                 $2, $8, $4*100, ($4+$5)*100, $2, $4, $4+$5}' \
                data/callhome/fullref.rttm | sort \
                | grep -v "iait_A_0035072_0035072" \
                > data/${dset}_${spk_num}/segments
            utils/fix_data_dir.sh data/${dset}_${spk_num}
            # Speaker ID is '[recid]_[speakerid]_0
            awk '{split($1,A,"_"); printf "%s %s_%s_0\n", $1, A[1], A[2]}' \
                data/${dset}_${spk_num}/segments > data/${dset}_${spk_num}/utt2spk
            LC_ALL=C utils/fix_data_dir.sh data/${dset}_${spk_num}
            # Generate rttm files for scoring
            steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
                data/${dset}_${spk_num}/utt2spk data/${dset}_${spk_num}/segments \
                data/${dset}_${spk_num}/rttm
            utils/data/get_reco2dur.sh data/${dset}_${spk_num}
            # Compose data/eval/${dset}_${spk_num}
            dst_dset=data/eval/${dset}_${spk_num}
            if ! validate_data_dir.sh --no-text --no-feats $dst_dset; then
                utils/copy_data_dir.sh data/${dset}_${spk_num} $dst_dset
                cp data/${dset}_${spk_num}/rttm $dst_dset/rttm
                awk -v dstdir=wav/eval/${dset}_${spk_num} '{print $1, dstdir"/"$1".wav"}' \
                    data/${dset}_${spk_num}/wav.scp > $dst_dset/wav.scp
                mkdir -p wav/eval/${dset}_${spk_num}
                wav-copy scp:data/${dset}_${spk_num}/wav.scp scp:$dst_dset/wav.scp
                utils/data/get_reco2dur.sh $dst_dset
                LC_ALL=C utils/fix_data_dir.sh $dst_dset
            fi
        fi
    done
    done
fi

set -eu
# Parse the config file to set bash variables like: $infer0_frame_shift, $infer1_subsampling
eval `yaml2bash.py --prefix infer0 $infer0_config`
eval `yaml2bash.py --prefix infer1 $infer1_config`

# Append gpu reservation flag to the queuing command
train_cmd+=" --gpu 1"
save_spkv_lab_cmd+=" --gpu 1"
infer_cmd+=" --gpu 1"

# Build directry names for an experiment
#  - Training
#     exp/diarize/model/${train_id}.${valid_id}.${train_config_id}
#  - Adapation from non-adapted averaged model
#     exp/diarize/model/${train_id}.${valid_id}.${train_config_id}/${ave_id}.${adapt_id}.${adapt_config_id}
#  - Inference
#     exp/diarize/infer/${train_id}.${valid_id}.${train_config_id}/${ave_id}.${adapt_id}.${adapt_config_id}.${adapt_ave_id}.${infer0_config_id}
#     exp/diarize/infer/${train_id}.${valid_id}.${train_config_id}/${ave_id}.${adapt_id}.${adapt_config_id}.${adapt_ave_id}.${infer1_config_id}
#  - Scoring
#     exp/diarize/scoring/${train_id}.${valid_id}.${train_config_id}/${ave_id}.${adapt_id}.${adapt_config_id}.${adapt_ave_id}.${infer0_config_id}
#     exp/diarize/scoring/${train_id}.${valid_id}.${train_config_id}/${ave_id}.${adapt_id}.${adapt_config_id}.${adapt_ave_id}.${infer1_config_id}
train_id=$(basename $train_set)
valid_id=$(basename $valid_set)
train_config_id=$(echo $train_config | sed -e 's%conf/%%' -e 's%/%_%' -e 's%\.yaml$%%')
save_spkv_lab_config_id=$(echo $save_spkv_lab_config | sed -e 's%conf/%%' -e 's%/%_%' -e 's%\.yaml$%%')
adapt_config_id=$(echo $adapt_config | sed -e 's%conf/%%' -e 's%/%_%' -e 's%\.yaml$%%')
infer0_config_id=$(echo $infer0_config | sed -e 's%conf/%%' -e 's%/%_%' -e 's%\.yaml$%%')
infer1_config_id=$(echo $infer1_config | sed -e 's%conf/%%' -e 's%/%_%' -e 's%\.yaml$%%')

# Additional arguments are added to config_id
train_config_id+=$(echo $train_args | sed -e 's/\-\-/_/g' -e 's/=//g' -e 's/ \+//g')
save_spkv_lab_config_id+=$(echo $save_spkv_lab_args | sed -e 's/\-\-/_/g' -e 's/=//g' -e 's/ \+//g')
adapt_config_id+=$(echo $adapt_args | sed -e 's/\-\-/_/g' -e 's/=//g' -e 's/ \+//g')
infer0_config_id+=$(echo $infer_args | sed -e 's/\-\-/_/g' -e 's/=//g' -e 's/ \+//g')
infer1_config_id+=$(echo $infer_args | sed -e 's/\-\-/_/g' -e 's/=//g' -e 's/ \+//g')

model_id=$train_id.$valid_id.$train_config_id
model_dir=exp/diarize/model/$model_id

if [ $stage -le 1 ]; then
    echo -e "\n==== stage 1: training model with simulated mixtures ===="
    # To speed up the training process, we first calculate input features
    # to NN and save shuffled feature data to the disc. During training,
    # we simply read the saved data from the disc.
    # Note: shuffled feature data (default total size: 336GB) are saved at the following place
    # exp/diarize/model/${train_id}.${valid_id}.${train_config_id}/data/
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

adapt_id=$(basename $adapt_set)
adapt_model_dir=exp/diarize/model/$model_id/$ave_id.$adapt_id.$adapt_config_id
save_spkv_lab_dir=$adapt_model_dir/$save_spkv_lab_config_id
if [ $stage -le 3 ]; then
    echo -e "\n==== stage 3: adapting model to CALLHOME dataset ===="

    # stage 3-1: saving speaker vector with label and speaker ID conversion table for initializing embedding dictionary
    echo "adapt_set: $(basename $adapt_set)"
    work=$save_spkv_lab_dir/.work
    mkdir -p $work
    $save_spkv_lab_cmd $work/save_spkv_lab.log \
        save_spkv_lab.py \
        -c $save_spkv_lab_config \
        $save_spkv_lab_args \
        $adapt_set \
        $model_dir/checkpoints/$ave_id.nnet.pth \
        $save_spkv_lab_dir \
        || exit 1
    echo -e "finished saving speaker vector with label"

    # stage 3-2: adapting model to CALLHOME dataset
    echo "adapting model at $adapt_model_dir"
    if [ -d $adapt_model_dir/checkpoints ]; then
        echo "$adapt_model_dir/checkpoints already exists. "
        echo " if you want to retry, please remove it."
        exit 1
    fi
    work=$adapt_model_dir/.work
    mkdir -p $work
    $train_cmd $work/train.log \
        train.py \
            -c $adapt_config \
            $adapt_args \
            --initmodel $model_dir/checkpoints/$ave_id.nnet.pth \
            --spkv-lab $save_spkv_lab_dir/spkvec_lab.npz \
            $adapt_set $adapt_valid_set $adapt_model_dir \
            || exit 1
fi

adapt_ave_id=avg${adapt_average_start}-${adapt_average_end}
if [ $stage -le 4 ]; then
    echo -e "\n==== stage 4: averaging adapted models ===="
    echo "averaging models into $adapt_model_dir/checkpoints/$adapt_ave_id.nnet.pth"
    if [ -s $adapt_model_dir/checkpoints/$adapt_ave_id.nnet.pth ]; then
        echo "$adapt_model_dir/checkpoints/$adapt_ave_id.nnet.pth already exists."
    fi
    last_epoch=$(ls $adapt_model_dir/checkpoints/ckpt_[0-9]*.pth | grep -v "/ckpt_0.pth"$ | wc -l)
    echo -e "last epoch of existence : $last_epoch"
    if [ $last_epoch -lt $adapt_average_end ]; then
        echo -e "error : adapt_average_end $adapt_average_end is too large."
        exit 1
    fi
    models=$(ls $adapt_model_dir/checkpoints/ckpt_[0-9]*.pth -tr | head -n $((${adapt_average_end}+1)) | tail -n $((${adapt_average_end}-${adapt_average_start}+1)))
    echo -e "take the average with the following models:"
    echo -e $models | tr " " "\n"
    model_averaging.py $adapt_model_dir/checkpoints/$adapt_ave_id.nnet.pth $models || exit 1
fi

infer_dir=exp/diarize/infer/$model_id/$ave_id.$(basename $adapt_set).$adapt_config_id.$adapt_ave_id.$infer0_config_id
if [ $stage -le 5 ]; then
    echo -e "\n==== stage 5: inference for evaluation (speaker counting: oracle) ===="
    echo "inference at $infer_dir"
    if [ -d $infer_dir ]; then
        echo "$infer_dir already exists. "
    fi
    for dset in ${test_dsets[@]}; do
        work=$infer_dir/$dset/.work
        mkdir -p $work
        $infer_cmd $work/infer.log \
            infer.py -c $infer0_config \
            $infer_args \
            data/eval/${dset} \
            $adapt_model_dir/checkpoints/$adapt_ave_id.nnet.pth \
            $infer_dir/$dset \
            || exit 1
    done
fi

scoring_dir=exp/diarize/scoring/$model_id/$ave_id.$(basename $adapt_set).$adapt_config_id.$adapt_ave_id.$infer0_config_id
if [ $stage -le 6 ]; then
    echo -e "\n==== stage 6: scoring for evaluation (speaker counting: oracle) ===="
    echo "scoring at $scoring_dir"
    if [ -d $scoring_dir ]; then
        echo "$scoring_dir already exists. "
    fi
    for dset in ${test_dsets[@]}; do
        work=$scoring_dir/$dset/.work
        mkdir -p $work
        find $infer_dir/$dset -iname "*.h5" > $work/file_list_$dset
        for med in 1 11; do
        for th in 0.3 0.4 0.5 0.6 0.7; do
            make_rttm.py --median=$med --threshold=$th \
                --frame_shift=$infer0_frame_shift --subsampling=$infer0_subsampling --sampling_rate=$infer0_sampling_rate \
                $work/file_list_$dset $scoring_dir/$dset/hyp_${th}_$med.rttm
            md-eval.pl -c 0.25 \
                -r data/eval/$dset/rttm \
                -s $scoring_dir/$dset/hyp_${th}_$med.rttm > $scoring_dir/$dset/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
        done
        done
        best_score.sh $scoring_dir/$dset
    done
fi

infer_dir=exp/diarize/infer/$model_id/$ave_id.$(basename $adapt_set).$adapt_config_id.$adapt_ave_id.$infer1_config_id
if [ $stage -le 7 ]; then
    echo -e "\n==== stage 7: inference for evaluation (speaker counting: estimated) ===="
    echo "inference at $infer_dir"
    if [ -d $infer_dir ]; then
        echo "$infer_dir already exists. "
    fi
    for dset in ${test_dsets[@]}; do
        work=$infer_dir/$dset/.work
        mkdir -p $work
        $infer_cmd $work/infer.log \
            infer.py -c $infer1_config \
            $infer_args \
            data/eval/${dset} \
            $adapt_model_dir/checkpoints/$adapt_ave_id.nnet.pth \
            $infer_dir/$dset \
            || exit 1
    done
fi

scoring_dir=exp/diarize/scoring/$model_id/$ave_id.$(basename $adapt_set).$adapt_config_id.$adapt_ave_id.$infer1_config_id
if [ $stage -le 8 ]; then
    echo -e "\n==== stage 8: scoring for evaluation (speaker counting: estimated) ===="
    echo "scoring at $scoring_dir"
    if [ -d $scoring_dir ]; then
        echo "$scoring_dir already exists. "
    fi
    for dset in ${test_dsets[@]}; do
        work=$scoring_dir/$dset/.work
        mkdir -p $work
        find $infer_dir/$dset -iname "*.h5" > $work/file_list_$dset
        for med in 1 11; do
        for th in 0.3 0.4 0.5 0.6 0.7; do
            make_rttm.py --median=$med --threshold=$th \
                --frame_shift=$infer1_frame_shift --subsampling=$infer1_subsampling --sampling_rate=$infer1_sampling_rate \
                $work/file_list_$dset $scoring_dir/$dset/hyp_${th}_$med.rttm
            md-eval.pl -c 0.25 \
                -r data/eval/$dset/rttm \
                -s $scoring_dir/$dset/hyp_${th}_$med.rttm > $scoring_dir/$dset/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
        done
        done
        best_score.sh $scoring_dir/$dset
    done
fi

echo "Finished !"
