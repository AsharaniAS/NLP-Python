# Training the model

from transformers import Trainer, TrainingArguments # type: ignore

class ModelTrainer:
    def __init__(self, model, tokenized_datasets, output_dir='./results', batch_size=8, num_epochs=3, weight_decay=0.01):
        self.model = model
        self.tokenized_datasets = tokenized_datasets
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay

    def get_training_args(self):
        return TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy='epoch',
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=self.weight_decay,
        )

    def train_model(self):
        training_args = self.get_training_args()
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_datasets['train'],
            eval_dataset=self.tokenized_datasets['test']
        )
        trainer.train()
        self.save_model(trainer)
        return trainer
    
    def save_model(self, trainer):
        trainer.save_model(self.output_dir)
