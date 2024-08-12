# Preprocess of the data
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer

# Download the nltk data
nltk.download('punkt')
nltk.download('stopwords')  
nltk.download('wordnet')

class PreprocessorUtils:
    def __init__(self):
        # Initialize NLP tools
        self.stop_words = set(stopwords.words('english'))   
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load('en_core_web_sm')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def word_token(self,text):
        # Tokenize the text
        return word_tokenize(text)

    def remove_stopwords(self,tokens):
        # Removal of stopwords
        tokens = [word for word in tokens if word.lower() not in self.stop_words]   
        return (tokens)

    def lemmatize(self,tokens):
        # Lemmatization of the words
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return (" ".join(tokens))
    
    def tokenize_function(self, examples):
        # Tokenize the text in examples
        return self.tokenizer(examples['text'], padding="max_length", truncation=True)
    
    def preprocess_text(self, text):
        text = self.word_token(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize(text)
        return (text)