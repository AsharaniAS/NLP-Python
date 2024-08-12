# NLP Model Pipeline

This project demonstrates an end-to-end NLP model pipeline using Python and the Hugging Face `transformers` library. The pipeline includes data loading, preprocessing, model training, evaluation, and saving the results.

## Folder Structure

- **dataset.py**: Contains `DatasetLoader` class for loading datasets.
- **preprocess_utils.py**: Contains `PreprocessUtils` class with utility functions for preprocessing.
- **preprocess_data.py**: Contains `DataPreprocessor` class that uses `PreprocessUtils` for preprocessing and tokenizing data.
- **model.py**: Contains `ModelLoader` class for loading the model.
- **train.py**: Contains `ModelTrainer` class for training the model.
- **evaluate.py**: Contains `ModelEvaluator` class for evaluating the model.
- **main.py**: Coordinates the entire workflow, handling exceptions.
- **requirements.txt**: Lists all the dependencies required to run the project.
- **results/**: Directory where the trained model and evaluation results are saved.

## Setup Instructions

1.Create a Virtual Environment:
- `python -m venv venv`
- `source venv/bin/activate`
- On Windows use: `venv\Scripts\activate`

2.Install Dependencies:
- `pip install -r requirements.txt`

3.Running project:
- `python main.py`

## File Descriptions:
a. dataset.py: DatasetLoader class: Loads the dataset using Hugging Face datasets library
b. preprocess_utils.py: PreprocessUtils class: Provides utility functions for text preprocessing.
c. preprocess_data.py: DataPreprocessor class: Uses PreprocessUtils to preprocess and tokenize the dataset.
d. main.py: This is main file.
e. model.py: ModelLoader class: Loads the model for sequence classification.
f. evaluate.py: Performs the model evalutation
g. train.py: Performs the Model training 
h. requirements.txt: "pip install -r requirements.txt"
    transformers: The main library for working with pre-trained models from Hugging Face.
    datasets: For loading and processing datasets.
    torch: PyTorch, a deep learning framework required for the models.
    tokenizers: Fast tokenization library used by Hugging Face transformers.
    numpy: For numerical operations.
    pandas: For data manipulation and analysis.
    scikit-learn: For additional machine learning utilities.

