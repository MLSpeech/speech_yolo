# SpeechYOLO: Detection and Localization of Speech Objects

Yael Segal (segal.yael@campus.technion.ac.il)\
Tzeviya Sylvia Fuchs (fuchstz@cs.biu.ac.il) \
Joseph Keshet (jkeshet@technion.ac.il)             


SpeechYOLO, inspired by the [YOLO](https://arxiv.org/pdf/1506.02640.pdf) algorithm , uses object detection methods from the vision domain for speech recognition. The goal of SpeechYOLO is to localize boundaries of utterances within the input signal, and to correctly classify them. Our system is composed of a convolutional neural network, with a simple least-meansquares loss function.


The paper can be found [here](https://arxiv.org/pdf/1904.07704.pdf). \
If you find our work useful, please cite: 
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

- Python 3.6+ 

- Pytorch 1.3.1

- numpy

- librosa

- soundfile

- Download the code:
    ```
    git clone https://github.com/MLSpeech/speech_yolo.git
    ```


## How to use

### Pretrain network on the Google Commands dataset:

- Download data from [Google Commands](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz).

- Each directory contains ```.wav``` files, each containing a single word. Split the keyword directories into ```train, val``` and ```test``` folders ([code](https://github.com/adiyoss/GCommandsPytorch/blob/master/make_dataset.py)). Your data should look as follows:

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
    You should have 30 folders (keywords) in every ```train \ val \test``` directory. See example in [gcommand_toy_example](https://github.com/MLSpeech/speech_yolo/tree/master/gcommand_toy_example).

- run:
	```
	python pretrain_run.py --train_path [path_to_data\train_folder]  
	                       --valid_path [path_to_data\val_folder] 
	                       --test_path [path_to_data\test_folder] 
	                       --arc VGG19 
	                       --cuda  
	                       --save_folder [directory_for_saving_models]  
	```
	
	This code runs a convolutional network for multiclass command classification. 

	Our pretraining model could be found [here](https://github.com/MLSpeech/speech_yolo/tree/master/gcommand_pretraining_model).

### Run SpeechYOLO

- We ran SpeechYOLO on the LibriSpeech dataset. See [data preparation](https://github.com/MLSpeech/speech_yolo/blob/master/librispeech_data_preparation.md) instructions.

- For simplicity, the SpeechYOLO code assumes that the `.wav` files are of length 1 sec each. 


- To train, run: 
	```
    python run_speech_yolo.py  --train_data [path_to_train_data]  
                               --val_data [path_to_validation_data]
                               --arc VGG19
                               --prev_classification_model [path_to_model_from_pretrain_part]
                               --save_folder [folder_to_save_speechyolo_model]
    ```

	If you want to load a previously trained `speech_yolo_model` file for further training, add:

	`--trained_yolo_model [path_to_file]`.

	Our trained model could be found here: [vgg11](https://drive.google.com/file/d/1XTBqeo-wf814UtCsF-OxmNJQiEfRznJ9/view), [vgg19](https://drive.google.com/file/d/1mkOn61zMzHi9S4XNhfDxnNSuV57OnoyN/view).

- To test, run:
	```
    python test_yolo.py  --train_data [path_to_train_data]  
                         --test_data [path_to_test_data]
                         --model [path_to_speechyolo_model]
    ```


Our results for threshold theta = 0.4 are:

````
threshold: 0.4
Actual Accuracy (Val): 0.7746057716719794
F1 regular mean: 0.806897659409815
precision: 0.836339972153278
recall: 0.7794578126322322
````



