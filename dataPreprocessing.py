import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


def load_data(path):
    return pd.read_csv(path)


def clean_data(data):
    # lowercase all text
    data['text'] = data['text'].str.lower()
    # remove punctuation signs, URL's, other symbols
    data['text'] = data['text'].str.replace(r'[^a-zA-Z0-9\s]', '')
    # remove stop words
    stop = stopwords.words('english')
    data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


def tokenize(data):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(data['text'].values)
    sequences = tokenizer.texts_to_sequences(data['text'].values)
    text = pad_sequences(sequences, maxlen=50)
    return text


def split(data):
    return train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
