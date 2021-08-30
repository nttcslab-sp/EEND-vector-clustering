# Environment
- GPU: GeForce GTX 1070
- CUDA Version: 10.1

# Main common conditions
- clustering algorithm using for inference: constrained AHC
- chunk size for adaptation and inference: 30 seconds

# Final DER

## Speaker counting: oracle
```
exp/diarize/scoring/swb_sre_tr_ns3_beta10_100000.swb_sre_cv_ns3_beta10_500.train/avg91-100.callhome1_spkall.adapt.avg21-25.infer_est_nspk0/callhome2_spk2/result_th0.5_med11_collar0.25: OVERALL SPEAKER DIARIZATION ERROR = 8.08 percent of scored speaker time  `(ALL)
exp/diarize/scoring/swb_sre_tr_ns3_beta10_100000.swb_sre_cv_ns3_beta10_500.train/avg91-100.callhome1_spkall.adapt.avg21-25.infer_est_nspk0/callhome2_spk3/result_th0.4_med11_collar0.25: OVERALL SPEAKER DIARIZATION ERROR = 11.27 percent of scored speaker time  `(ALL)
exp/diarize/scoring/swb_sre_tr_ns3_beta10_100000.swb_sre_cv_ns3_beta10_500.train/avg91-100.callhome1_spkall.adapt.avg21-25.infer_est_nspk0/callhome2_spk4/result_th0.4_med11_collar0.25: OVERALL SPEAKER DIARIZATION ERROR = 15.01 percent of scored speaker time  `(ALL)
exp/diarize/scoring/swb_sre_tr_ns3_beta10_100000.swb_sre_cv_ns3_beta10_500.train/avg91-100.callhome1_spkall.adapt.avg21-25.infer_est_nspk0/callhome2_spk5/result_th0.5_med11_collar0.25: OVERALL SPEAKER DIARIZATION ERROR = 23.14 percent of scored speaker time  `(ALL)
exp/diarize/scoring/swb_sre_tr_ns3_beta10_100000.swb_sre_cv_ns3_beta10_500.train/avg91-100.callhome1_spkall.adapt.avg21-25.infer_est_nspk0/callhome2_spk6/result_th0.5_med11_collar0.25: OVERALL SPEAKER DIARIZATION ERROR = 26.56 percent of scored speaker time  `(ALL)
exp/diarize/scoring/swb_sre_tr_ns3_beta10_100000.swb_sre_cv_ns3_beta10_500.train/avg91-100.callhome1_spkall.adapt.avg21-25.infer_est_nspk0/callhome2_spkall/result_th0.4_med11_collar0.25: OVERALL SPEAKER DIARIZATION ERROR = 12.22 percent of scored speaker time  `(ALL)
```

## Speaker counting: estimated
```
exp/diarize/scoring/swb_sre_tr_ns3_beta10_100000.swb_sre_cv_ns3_beta10_500.train/avg91-100.callhome1_spkall.adapt.avg21-25.infer_est_nspk1/callhome2_spk2/result_th0.5_med11_collar0.25: OVERALL SPEAKER DIARIZATION ERROR = 7.96 percent of scored speaker time  `(ALL)
exp/diarize/scoring/swb_sre_tr_ns3_beta10_100000.swb_sre_cv_ns3_beta10_500.train/avg91-100.callhome1_spkall.adapt.avg21-25.infer_est_nspk1/callhome2_spk3/result_th0.4_med11_collar0.25: OVERALL SPEAKER DIARIZATION ERROR = 11.93 percent of scored speaker time  `(ALL)
exp/diarize/scoring/swb_sre_tr_ns3_beta10_100000.swb_sre_cv_ns3_beta10_500.train/avg91-100.callhome1_spkall.adapt.avg21-25.infer_est_nspk1/callhome2_spk4/result_th0.4_med11_collar0.25: OVERALL SPEAKER DIARIZATION ERROR = 16.38 percent of scored speaker time  `(ALL)
exp/diarize/scoring/swb_sre_tr_ns3_beta10_100000.swb_sre_cv_ns3_beta10_500.train/avg91-100.callhome1_spkall.adapt.avg21-25.infer_est_nspk1/callhome2_spk5/result_th0.5_med11_collar0.25: OVERALL SPEAKER DIARIZATION ERROR = 21.21 percent of scored speaker time  `(ALL)
exp/diarize/scoring/swb_sre_tr_ns3_beta10_100000.swb_sre_cv_ns3_beta10_500.train/avg91-100.callhome1_spkall.adapt.avg21-25.infer_est_nspk1/callhome2_spk6/result_th0.5_med11_collar0.25: OVERALL SPEAKER DIARIZATION ERROR = 23.10 percent of scored speaker time  `(ALL)
exp/diarize/scoring/swb_sre_tr_ns3_beta10_100000.swb_sre_cv_ns3_beta10_500.train/avg91-100.callhome1_spkall.adapt.avg21-25.infer_est_nspk1/callhome2_spkall/result_th0.4_med11_collar0.25: OVERALL SPEAKER DIARIZATION ERROR = 12.49 percent of scored speaker time  `(ALL)
```
