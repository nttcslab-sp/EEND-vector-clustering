# EEND-vector clustering

The EEND-vector clustering (End-to-End-Neural-Diarization-vector clustering) is a speaker diarization framework that integrates two complementary major diarization approaches, i.e., traditional clustering-based and emerging end-to-end neural network-based approaches, to make the best of both worlds. In [1] it is shown that the EEND-vector clustering outperforms EEND when the recording is long (e.g., more than 5 min), while in [2] it is shown based on CALLHOME data that it outperforms x-vector clustering and EEND-EDA especially when the number of speakers in recordings is large.

This repository contains an example implementation of the EEND-vector clustering based on Pytorch to reproduce the results in [2], i.e., the CALLHOME experiments. For the trainer, we use [Padertorch](https://github.com/fgnt/padertorch). This repository is implemented based on [EEND](https://github.com/hitachi-speech/EEND) and relies on some useful functions provided therein.
 

## References
[1] Keisuke Kinoshita, Marc Delcroix, and Naohiro Tawara, "Integrating end-to-end neural and clustering-based diarization: Getting the best of both worlds," Proc. ICASSP, pp. 7198–7202, 2021

[2] Keisuke Kinoshita, Marc Delcroix, and Naohiro Tawara, "Advances in integration of end-to-end neural and clustering-based diarization for real conversational speech," Proc. Interspeech, 2021 (to appear)

## Citation
```
@inproceedings{eend-vector-clustering,
 author = {Keisuke Kinoshita and Marc Delcroix and Naohiro Tawara},
 title = {Integrating End-to-End Neural and Clustering-Based Diarization: Getting the Best of Both Worlds},
 booktitle = {{ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}},
 pages={7198-7202}
 year = {2021}
}
```

## Install tools
### Requirements
 - NVIDIA CUDA GPU
 - CUDA Toolkit (version == 9.2, 10.1 or 10.2)

### Install kaldi and python environment
```bash
cd tools
make
```
- This command builds kaldi at `tools/kaldi`
  - if you want to use pre-build kaldi
    ```bash
    cd tools
    make KALDI=<existing_kaldi_root>
    ```
    This option make a symlink at `tools/kaldi`
- This command extracts miniconda3 at `tools/miniconda3`, and creates conda envirionment named 'eend'
- Then, installs Pytorch and Padertorch into 'eend' environment
  - use CUDA in `/usr/local/cuda/`
    - if you need to specify your CUDA path
      ```bash
      cd tools
      make CUDA_PATH=/your/path/to/cuda-10.1
      ```
      The pytorch install command to be executed is depended on your CUDA version.
      See https://pytorch.org/get-started/previous-versions/
- Then, clones [EEND](https://github.com/hitachi-speech/EEND) to reference symbolic links stored under `eend/`, `egs/` and `utils/`

## Test recipe (mini_librispeech)
### Configuration
- Modify `egs/mini_librispeech/v1/cmd.sh` according to your job schedular.
If you use your local machine, use "run.pl" (default).
If you use Grid Engine, use "queue.pl"
If you use SLURM, use "slurm.pl".
For more information about cmd.sh see http://kaldi-asr.org/doc/queue.html.
### Run data preparation, training, inference, and scoring
```bash
cd egs/mini_librispeech/v1
CUDA_VISIBLE_DEVICES=0 ./run.sh
```
- See `RESULT.md` and compare with your result.

## CALLHOME experiment
### Configuraition
- Modify `egs/callhome/v1/cmd.sh` according to your job schedular.
If you use your local machine, use "run.pl" (default).
If you use Grid Engine, use "queue.pl"
If you use SLURM, use "slurm.pl".
For more information about cmd.sh see http://kaldi-asr.org/doc/queue.html.
### Run data preparation, training, inference, and scoring
```bash
cd egs/callhome/v1
CUDA_VISIBLE_DEVICES=0 ./run.sh --db_path <db_path>
# <db_path> means absolute path of the directory where the necessary LDC corpora are stored.
```
- See `RESULT.md` and compare with your result.
- If you want to run multi-GPU training, simply set `CUDA_VISIBLE_DEVICES` appropriately. This environment variable may be automatically set by your job schedular such as SLURM.

