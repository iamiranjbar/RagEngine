import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

class TextPreprocessor:
    def __init__(self):
        nltk.download('punkt_tab', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
    
    def process(self, text, stemming=False):
        tokens = word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in self.stop_words]
        if stemming:
            processed = [self.stemmer.stem(word) for word in filtered_tokens]
        else:
            processed = [self.lemmatizer.lemmatize(word) for word in filtered_tokens]
        return processed
