# Detection of LLM-generated Text - Natural Language Processing, Tel Aviv University, Spring 2024

This project consists of designing a model for detecting llm-generated text by transforming existing text datasets via llm tasks and fine-tuning a pretrained llm for sequence classification, and was done as part of the Natural Language Processing course at Tel Aviv University, Spring 2024, under the instruction of Maor Ivgi.

## Environment Setup
1. [Install Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. `conda env create -f env.yml`
3. `conda activate llm_project`
4. [Install Pytorch](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

## Running Tasks
For running tasks use `python main.py --task {task}`, the following tasks are supported:
* `preprocess` - generate an LLM-transformed dataset from a source text dataset and a source LLM
* `persist_to_csv` - save a preprocessed dataset in a csv file at the dataset's folder and the folder `csv` (automatically done in the `preprocess` task)
* `train` - train a classifier on a preprocess dataset (requires the `preprocess` task done before)
* `evaluare` - evaluate a classifier on an evaluation dataset
* `interpret` - save the attribution visualization of the classification of an example
* `test` - for debugging purposes, allows to use an LLM for text-generation purposes
### Arguments
See the [args.py](./args.py) file for the full list of arguments that can be passed, each task may use any subset of arguments for its execution.
