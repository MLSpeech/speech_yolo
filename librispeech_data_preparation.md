## LibriSpeech Data Preparation


The librispeech dataset can be found [here](http://www.openslr.org/12). The data is already divided to train, validation and test sets. We used both the `clean` and `other` segments for training, but tested them separately. 

The dataset is given with `.wav` and `.lab` files, where the `.lab` files are transcriptions. In order to obtain word alignemnts, we used the Montreal Forced Aligner ([MFA](https://montreal-forced-aligner.readthedocs.io/en/latest/)). 

For convenience, we trimmed the `.wav` files to be 1 second each. Every `.wav` file has a corresponding `.wrd` file. A `.wrd` file looks as follows:

```
8703 13823 able 
63 5663 those 
5663 7743 who 
7743 8703 are 

```

That is, every line contains `(start, end , word)`, which denote the start and end times (in frames) for a given word. The file only contains words that are included within a predefined list of keywords. We chose the top 1000 words that were uttered in Librispeech (ignoring stop words like "the", "a"). The list of words we chose could be found [here](https://github.com/MLSpeech/speech_yolo/blob/master/word_list.txt).


Your data should look as follows:

```
data
    └───train
    |   |_____word_1
    │   |       │   1.wav
    │   |       │   1.wrd
    │   |       │   2.wav
    │   |       │   2.wrd
    │   |
    |   |_____word_2
    │   |       │   4.wav
    │   |       │   4.wrd
    │   |       │   6.wav   
    │   |       │   6.wrd       
    └───val
    |   |_____word_1
    │   |       │   7.wav
    │   |       │   7.wrd
    │   |       │   8.wav
    │   |       │   8.wrd
    │   |
    |   |_____word_2
    │   |       │   10.wav
    │   |       │   10.wrd
    │   |       │   11.wav
    │   |       │   11.wrd     
    └───test
    |   |_____word_1
    │   |       │   13.wav
    │   |       │   13.wrd
    │   |       │   14.wav
    │   |       │   14.wrd
    │   |
    |   |_____word_2
    │   |       │   16.wav
    │   |       │   16.wrd
    │   |       │   17.wav
    │   |       │   17.wrd     
```

See [toy example](https://github.com/MLSpeech/speech_yolo/tree/master/librispeech_toy_example).

