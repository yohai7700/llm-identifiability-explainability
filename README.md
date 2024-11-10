# LLM Identifiability Explainability

This project consists of designing a model for detecting llm-generated text by transforming existing text datasets via llm tasks and fine-tuning a pretrained llm for sequence classification.

## Environment Setup
1. [Install Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. `conda env create -f env.yml`
3. `conda activate llm_project`
4. [Install Pytorch](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

## Tasks
There are several tasks implemented in the `main.py` file, 'preprocess', 'persist_to_csv', 'train', 'test', 'predict', you can run each by `python main.py --task={task}`, e.g. `python main.py --task=predict`, 

the flow of our process is:
* by preprocessing the dataset which generates an llm-incrorporated dataset and persists it to the `data/checkpoints` folder
* `persist_to_csv` is used when you wish to visually see the dataset values
* training a classification model on the preprocessed dataset
* evaluating the classification model on different datasets via the `test` task
