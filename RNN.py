from keras.models import Sequential
from keras import layers
import dataPreprocessing

# Hyperparameters
max_features = 1500
max_length = 150
epochs = 2

# Preprocessing Steps:
read_data = dataPreprocessing.load_data('data/processed_data.csv')
dataPreprocessing.clean_data(read_data)
x_train, x_test, y_train, y_test = dataPreprocessing.split(read_data)
x_train, x_test = dataPreprocessing.tokenize(x_train, x_test, max_features, max_length)
print('Preprocessing complete.')

# create empty model
model = Sequential()
# add embedding layer
model.add(layers.Embedding(max_features, 150, input_length=max_length))
# add LSTM layer
model.add(layers.LSTM(150, dropout=0.2, recurrent_dropout=0.2))
# add dense layer
model.add(layers.Dense(1, activation='sigmoid'))

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Model Compiled")

# train model
model.fit(x_train, y_train, epochs=epochs, validation_split=0.2)
print("Model Trained")

# evaluate model
loss, accuracy = model.evaluate(x_test, y_test)
print('Accuracy: %f' % (accuracy*100))
print('Loss: %f' % (loss*100))
