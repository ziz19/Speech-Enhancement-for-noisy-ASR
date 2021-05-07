This is the project from graduate course Spoken Language Technology from UT Austin

The project mainly studies the use of speech enhancement module before ASR in noisy conditions

The dataset is LibriMix, and the model is trained and evaluated using SpeechBrain

There are three stages of this project:

1) Use a pretrained ASR model on noisy speech as Baseline
2) Train a speech enhancement module and use it as the front end for pretrained ASR
3) Fine-tune the pretrained ASR using enhanced speech generated from speech enhancement module

**Reference**

LibriMix
```
@misc{cosentino2020librimix,
    title={LibriMix: An Open-Source Dataset for Generalizable Speech Separation},
    author={Joris Cosentino and Manuel Pariente and Samuele Cornell and Antoine Deleforge and Emmanuel Vincent},
    year={2020},
    eprint={2005.11262},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
```

Speechbrain
```
@misc{SB2021,
    author = {Ravanelli, Mirco and Parcollet, Titouan and Rouhe, Aku and Plantinga, Peter and Rastorgueva, Elena and Lugosch, Loren and Dawalatabad, Nauman and Ju-Chieh, Chou and Heba, Abdel and Grondin, Francois and Aris, William and Liao, Chien-Feng and Cornell, Samuele and Yeh, Sung-Lin and Na, Hwidong and Gao, Yan and Fu, Szu-Wei and Subakan, Cem and De Mori, Renato and Bengio, Yoshua },
    title = {SpeechBrain},
    year = {2021},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\\url{https://github.com/speechbrain/speechbrain}},
  }
  ```