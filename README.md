ProxMaP
=======
This repository contains the code for [**ProxMaP**: **Prox**imal Occupancy **Ma**p **P**rediction for Efficient Indoor Robot Navigation](https://arxiv.org/abs/2203.04177) by [Vishnu D. Sharma](http://vishnuduttsharma.github.io/), [Jingxi Chen](http://https//codingrex.github.io/), and [Pratap Tokekar](http://tokekar.com/).

![ProxMaP Overview](http://raaslab.org/projects/ProxMaP/static/images/overview_ooccmap.png)


Data Generation
===============
Data is generated with [AI2THOR](https://ai2thor.allenai.org/) simulator. 
Create directories for saving data and change the paths at Line 14-16 in `data_generation.py`.
Then Run `python data_generation.py`. This will generate *.npy* files, and a desceription file named `updated_description_ang0.csv`, which will be used by dataloader.

## Download data
Already generated data is avaialable for download as a [zip file here](https://obj.umiacs.umd.edu/proxmap/ProxMaP_data.zip).


Training
========
For training the model, create directories named `saved_models` (for saving models as _.pth_ files) and `tblogdir` (for saving tensorboard logs). Then run the following command `python train_classification.py --epoch 500 --batch-size 8 --learning-rate 0.01 --validation 10 --loss-function 'crossent' --logdir ./tblogdir/`. 

You can look at other training parameters by running `python train_classification.py --help`.

## Pre-trained Model
Pretrained classification model is avialable for [download here](https://obj.umiacs.umd.edu/proxmap/classification_model.pth).

## Regression Model Training
The regression models can be run with similar command as above using the file named `train_regression.py`.


Testing
=======
For testing the model, run `python test_classification.py --model-path ./saved_models/classical_model.pth --batch-size 8 --device gpu --show` with appropriate model name.

## Regression Model Testing
The regression models can be run with similar command as above using the file named `testregression.py`.


File Description
================
## Data generation
*	data_generation.py: File to generate dataset. This directory already includes data, so you don't need to run it.
*	helper_v3.py: Helper code based on AI2THOR for data geenration
*	updated_description_ang0.csv: File containing metatdata for dataset (numpy files)

## Model training
*	train_classification.py: File to train the classification model
*	test_classification.py: File to test the classification model
*	train_regression.py: File to train the regression model
*	test_regression.py: File to test the regression model
*	commands.txt: Shows example training and test command

Citing
======
If you use our code or data, please cite our work with following:
```
@article{sharma2023proxmap,
  title={ProxMaP: Proximal Occupancy Map Prediction for Efficient Indoor Robot Navigation},
  author={Sharma, Vishnu Dutt and Chen, Jingxi and Tokekar, Pratap},
  journal={arXiv preprint arXiv:2305.05519},
  year={2023}
}
```


