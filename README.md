# Deep Reinforcement Learning for QDMR Parsing
##### authors:
* [Amir Matz](https://github.com/amirmatz/)
* [Yoav Halperin](https://github.com/yoavhh/)

This repository contains the code of the final project for the course [Natural Language Processing](http://www.cs.tau.ac.il/~joberant/teaching/nlp_fall_2019_2020/index.html) by [Jonathan Berant](http://www.cs.tau.ac.il/~joberant/).
  
This work was based on [BREAK It Down: A Question Understanding Benchmark](https://arxiv.org/pdf/2001.11770v1.pdf), 
and was trained using the dataset described in this paper, which you may find [here](https://allenai.github.io/Break/).

You can configure the path of the input dataset in `dataset_reader.py` in the `_mode_to_files` mapping.

## Installation
Installation instructions:  

    # $ conda create --name nlp_project --file nlp_project_env.txt

## Training
How to train the model:  

    # $ python3 ./train.py
  

## Evaluating
The training process automatically stores the model every 100 epochs.  
You can get the output of the model using: 

    # $ python3 ./train.py <epoch> <mode>
Whereas mode in one of: train / test / dev  
This will create a `results_<model>_<epoch>.pkl` output file which can be loaded using:

    # [1] import pandas as pd
    # [2] results = pd.read_pickle("<path_to_file>", compression="gzip")

### Previous versions
You may find  previous versions which use BLEU score here: [link](https://github.com/amirmatz/DQN/tags)