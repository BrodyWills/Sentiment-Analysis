import re
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
    data['text'] = data['text'].apply(lambda x: x.lower())
    # remove usernames tagged in tweets
    data['text'] = data['text'].apply(lambda x: re.sub('@[^\s]+', '', x))
    # remove symbols
    data['text'] = data['text'].apply((lambda x: re.sub('[^a-z0-9\s]', '', x)))
    # remove stop words
    stop = stopwords.words('english')
    data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


def tokenize(x_train, x_test, max_features, max_length):
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(x_train)
    train_sequences = tokenizer.texts_to_sequences(x_train)
    train = pad_sequences(train_sequences, maxlen=max_length)
    test_sequences = tokenizer.texts_to_sequences(x_test)
    test = pad_sequences(test_sequences, maxlen=max_length)
    return [train, test]


def split(data):
    return train_test_split(data['text'], data['sentiment'], test_size=0.2, random_state=42)


def summarize(data):
    print('Data Shape: ', data.shape)
    # print average length of tweets
    print('Average length of tweet: {}'.format(round(data['text'].str.len().mean())))
    # print number of positive and negative tweets
    print('Number of positive/negative tweets: {}'.format(data['sentiment'].value_counts()))
