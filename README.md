Text Classification Pytorch
===

Pytorch implementation of TextCNN and TextRNN for text classification task.


## Environment
- Python 3.6
- Pytorch 1.1.0
- Debugsummary 0.1
- Numpy 1.18.1


## Dataset
We use sentence polarity dataset v1.0 of [Movie Review Data](http://www.cs.cornell.edu/people/pabo/movie-review-data).


## Training TextCNN
```bash
python run_textcnn.py
```

## Training TextRNN
You can use either rnn cell, gru cell or lstm cell. You can use the biderectional version. Change the configurations in the code.
```bash
python run_textrnn.py
```


## References
This code is based on dennybritz's [cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf).
Many thanks!
