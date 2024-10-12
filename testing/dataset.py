from datasets import load_dataset, DatasetDict

imdb_dataset = load_dataset("shawhin/imdb-truncated")
yelp_dataset = load_dataset("Yelp/yelp_review_full")

dataset = ds = DatasetDict({
    'train': yelp_dataset['train'],
    'validation': yelp_dataset['test']
})