# Running the code

The code uses Python 3.8.10 which is the Python 3 version in the NUS SOC Compute Cluster. It would probably be useful to use an environment, such as through `virtualenv`, to run this code. The code assumes that the LUN dataset's csv files are in a folder called `Data` at the same level as the `src` folder.

Install the required libraries using `pip`. You can either simply run

`pip install -r requirements.txt`

or install `tensorflow-gpu`, `pandas`, `numpy`, `sklearn`, `scipy`, and `nltk`, one by one with `pip`.

To train and evaluate a model encapsulated by a wrapper, run

`python -m src.main --model_wrapper=[MODEL_WRAPPER_ID] --dataset=[DATASET_ID]`

Model wrapper and dataset IDs can be found in `src/model_wrappers/__init__.py` and `src/data/__init__.py` respectively.

Some other optional parameters for running the code are listed below:

- `--mode=[train | test | full]` is used to choose whether to only train, only test, or do both.
- `--save_filepath=[SAVE_FILEPATH]` is used to specify where to save the model to.
- `--load_filepath=[LOAD_FILEPATH]` is used to specify where to load the model from.

# Reproducing the results in the report

The model wrappers used have IDs `nltk_rnn` for the recurrent neural network and `attention` for the attention-based neural network. The more general Kaggle-BUPT-GAMMA dataset used to train and do an initial evaluation has the ID `kaggle_bupt_gamma` and while the Singaporean media dataset used to do the final evaluation has the ID `singapore_test`.

To train and evaluate the model on the Kaggle-BUPT-GAMMA dataset, run

`python -m src.main --model_wrapper=[MODEL_WRAPPER_ID] --dataset=kaggle_bupt_gamma --save_filepath=[FILEPATH]`

and to evaluate it again on the Singapore media dataset, run

`python -m src.main --mode=test --model_wrapper=[MODEL_WRAPPER_ID] --dataset=singapore_test --load_filepath=[FILEPATH]`

where `MODEL_WRAPPER_ID` are either `nltk_rnn` or `attention` and `FILEPATH` can be any suitable filepath to save and load the model.

For example,

`python -m src.main --model_wrapper=nltk_rnn --dataset=kaggle_bupt_gamma --save_filepath=checkpoints/nltk_rnn/checkpoint`

`python -m src.main --mode=test --model_wrapper=nltk_rnn --dataset=singapore_test --load_filepath=checkpoints/nltk_rnn/checkpoint`

or

`python -m src.main --model_wrapper=attention --dataset=kaggle_bupt_gamma --save_filepath=checkpoints/attention/checkpoint`

`python -m src.main --mode=test --model_wrapper=attention --dataset=singapore_test --load_filepath=checkpoints/attention/checkpoint`
