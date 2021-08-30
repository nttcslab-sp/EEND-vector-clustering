# Environment
- GPU: GeForce GTX 1070
- CUDA Version: 10.1

# Main common conditions
- clustering algorithm using for inference: constrained AHC
- chunk size for inference: 30 seconds

# Training curve (validation loss)
Note that validation loss := (1 - spk_loss_ratio) * pit_loss

```
grep "Mean loss" exp/diarize/model/train_clean_5_ns3_beta2_500.dev_clean_2_ns3_beta2_500.train/.work/train.log
Finished Validation. Mean loss: 0.6628542523635061
Finished Validation. Mean loss: 0.6607387272935165
Finished Validation. Mean loss: 0.6548653715535214
Finished Validation. Mean loss: 0.6455376411739149
Finished Validation. Mean loss: 0.6333835890418604
Finished Validation. Mean loss: 0.6193064156331514
Finished Validation. Mean loss: 0.6045306845715172
Finished Validation. Mean loss: 0.5902855377448233
Finished Validation. Mean loss: 0.5776547281365646
Finished Validation. Mean loss: 0.5674387065987838
Finished Validation. Mean loss: 0.5597694854987295
```

# Final DER

## Speaker counting: oracle
```
exp/diarize/scoring/train_clean_5_ns3_beta2_500.dev_clean_2_ns3_beta2_500.train.avg8-10.infer_est_nspk0/dev_clean_2_ns3_beta2_500/result_th0.5_med11_collar0.25: OVERALL SPEAKER DIARIZATION ERROR = 36.70 percent of scored speaker time  `(ALL)
```

## Speaker counting: estimated
```
exp/diarize/scoring/train_clean_5_ns3_beta2_500.dev_clean_2_ns3_beta2_500.train.avg8-10.infer_est_nspk1/dev_clean_2_ns3_beta2_500/result_th0.5_med11_collar0.25: OVERALL SPEAKER DIARIZATION ERROR = 41.48 percent of scored speaker time  `(ALL)
```
