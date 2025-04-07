# Investigation of Spatial Self-Supervised Learning and Its Application to Target Speaker Speech Recognition
This is a repository of guided neural fast full-rank spatial covariance analysis (guided neural FastFCA).


## Installation
```bash
pip install git+https://github.com/b-sigpro/neural-gfca.git
```

## Inference
Pre-trained models are available at the [release page](https://github.com/b-sigpro/neural-gfca/releases).
One utterance in a mixture recording `[src_file].wav` can be extracted to `[dst_file].wav` the following command.
```bash
python -m neural_gfca.separate one ./neural-gfca.16ch-qini-nsfsim.Ns=6/ [src_file].wav [dst_file].wav --target --n_mic=16 --drop_context --normalize=exceed --use_mvdr
```
The script automatically reads `[src_file].info`, which must be a Python pickle file of a dictionary with the following format:
```python
{
    "act": np.ndarray([T, N]),  # binary activations of N speakers, the 1st speaker (n=0) is the target.
    "start": int,  # start time sample of the target,
    "end": int,  # end time sample of the target,
}
```

If you have out of memory issue, you can use the following option:
```bash
task.encoder.diagonalizer._target_=neural_gfca.diagonalizers.iss_nrmxt_zhang3_cnt_fblk_diagonalizer.ISSDiagonalizer
```
This option diagonalizes the mixture for each block of frequency bins, which will takes less memory but more computational time.

## Reference
```bibtex
@inproceedings{bando2025investigation,
  title={Investigation of Spatial Self-Supervised Learning and Its Application to Target Speaker Speech Recognition},
  author={Yoshiaki Bando and Samuele Cornell and Satoru Fukayama and Shinji Watanabe},
  booktitle={IEEE ICASSP 2025},
  year={2025}
}
```

## Acknowledgement
This work is based on results obtained from a project, Programs for Bridging the gap between R&D and the IDeal society (society 5.0) and Generating Economic and social value (BRIDGE)/Practical Global Research in the AI × Robotics Services, implemented by the Cabinet Office, Government of Japan.
