# Importing the dataset 'imdb' from datasets
from datasets import load_dataset # type: ignore

class DatasetLoader:
    def __init__(self,dataset_name):
        self.dataset = dataset_name

    def load_dataset(self, dataset_name='imdb'):
        self.dataset = load_dataset(dataset_name)
        return self.dataset
