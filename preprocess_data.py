from preprocess_utils import PreprocessorUtils

# Preprocess of the data
class DataPreprocessor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.utils = PreprocessorUtils()

    def preprocess_and_tokenize(self, dataset):
        # calling Preprocess the  tokenize function on dataset
        preproceed_dataset = dataset.map(lambda examples: {'text': [self.utils.preprocess_text(text) for text in examples['text']]})
        tokenized_dataset = preproceed_dataset.map(self.utils.tokenize_function, batched=True)
        return tokenized_dataset