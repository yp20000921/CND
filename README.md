# CND

Source code for IJCAI 2023 paper: Efficient Sign Language Translation with a Curriculum-based Non-autoregressive Decoder

This code is based on [SignJoey](https://github.com/neccam/slt).
 
## Requirements

* [Optional] Create a conda or python virtual environment.

* Install required packages using the `requirements.txt` file.

    `pip install -r requirements.txt`

## Usage

We adopt the [SMKD](https://github.com/ycmin95/VAC_CSLR) to pretrain sign embeddings extract the visual features.

Then we train the model by

  `python -m signjoey train configs/sign.yaml` 

Note that you need to update the `data_path` parameters in your config file.

Then test the bleu score of the checkpoint

  `python -m signjoey test configs/sign.yaml --ckpt xxx` 

