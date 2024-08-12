# Main file 
from dataset import DatasetLoader
from preprocess_data import DataPreprocessor
from model import ModelLoader
from train import ModelTrainer
from evaluate import ModelEvaluator

class NLPModelPipeline:
    def __init__(self, dataset_name, model_name, output_dir='./results', num_labels=2, batch_size=8, num_epochs=3, weight_decay=0.01):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.output_dir = output_dir
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.dataset_loader = DatasetLoader(dataset_name)
        self.preprocessor = DataPreprocessor(dataset_name)
        self.model_loader = ModelLoader(model_name, num_labels)

    def load_and_preprocess_data(self):
        print("Loading and preprocessing data...")
        dataset = self.dataset_loader.load_dataset()
        tokenized_datasets = self.preprocessor.preprocess_and_tokenize(dataset)
        print("Data loaded and preprocessed successfully!")
        return tokenized_datasets
    
    def run(self):
        try:
            tokenized_datasets = self.load_and_preprocess_data()
            print("Loading model.....")
            model = self.model_loader.load_model()
            print("Model loaded successfully!")
            print("Starting training....")
            trainer = ModelTrainer(model, tokenized_datasets, self.output_dir, self.batch_size, self.num_epochs, self.weight_decay).train_model()
            print("Training completed!")
            print("Evaluating model....")
            eval_results = ModelEvaluator(trainer).evaluate_model()
            print("Evaluation completed! Results saved to eval_results.json file in results folder")
            print(eval_results)
        except Exception as e:
            print(f"An error occurred: {e}")
        
if __name__ == '__main__':
    model_pipeline = NLPModelPipeline(dataset_name='imdb', model_name='bert-base-uncased')
    model_pipeline.run()