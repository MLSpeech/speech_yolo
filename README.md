# SpeechYOLO: Detection and Localization of Speech Object

Yael Segal (segalya@cs.biu.ac.il)\
Tzeviya Sylvia Fuchs (fuchstz@cs.biu.ac.il) \
Joseph Keshet (joseph.keshet@cs.biu.ac.il)             


SpeechYOLO, inspired by the [YOLO](https://arxiv.org/pdf/1506.02640.pdf) algorithm , uses object detection methods from the vision domain for speech recognition. The goal of SpeechYOLO is to localize boundaries of utterances within the input signal, and to correctly classify them. Our system is composed of a convolutional neural network, with a simple least-meansquares loss function.


The paper can be found at (https://arxiv.org/pdf/1904.07704.pdf). \
If you find our work useful please cite : 
```
@article{segal2019speechyolo,
  title={SpeechYOLO: Detection and Localization of Speech Objects},
  author={Segal, Yael and Fuchs, Tzeviya Sylvia and Keshet, Joseph},
  journal={Proc. Interspeech 2019},
  pages={4210--4214},
  year={2019}
}
```

------


## Installation instructions

- Python 3.6+ (???)

- Download the code:
    ```
    git clone https://github.com/MLSpeech/speech_yolo.git
    ```
-TODO: finish.... 


## How to use: 

### Pretrain the network on the Google Commands dataset

- Download data from [Google Commands](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz).

- Split the ```.wav``` files into ```train, val``` and ```test``` folders ([code][https://github.com/adiyoss/GCommandsPytorch/blob/master/make_dataset.py]). Each file contains a single word. Your data should look as follows:

    ```
    data
	    └───train
	    |   |_____word_1
	    │   |       │   1.wav
	    │   |       │   2.wav
	    │   |       │   3.wav
	    │   |
	    |   |_____word_2
	    │   |       │   4.wav
	    │   |       │   5.wav
	    │   |       │   6.wav          
	    └───val
	    |   |_____word_1
	    │   |       │   7.wav
	    │   |       │   8.wav
	    │   |       │   9.wav
	    │   |
	    |   |_____word_2
	    │   |       │   10.wav
	    │   |       │   11.wav
	    │   |       │   12.wav     
	    └───test
	    |   |_____word_1
	    │   |       │   13.wav
	    │   |       │   14.wav
	    │   |       │   15.wav
	    │   |
	    |   |_____word_2
	    │   |       │   16.wav
	    │   |       │   17.wav
	    │   |       │   18.wav     
    ```
    You should have 30 folders (keywords) in every ```train \ val \test``` directory.

- run ```python pretrain_run.py --train_path [path_to_data\train_folder] 
								--valid_path [path_to_data\val_folder]
								--test_path [path_to_data\test_folder]
								--arc VGG19
								--cuda 
								--save_folder [directory_for_saving_models] ```

	This code runs a convolutional network for multiclass command classification.

### Run SpeechYOLO