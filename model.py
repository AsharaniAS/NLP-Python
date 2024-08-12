# model initialization
from transformers import AutoModelForSequenceClassification # type: ignore

class ModelLoader:
    def __init__(self, model_name, num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels

    def load_model(self):
        return AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
