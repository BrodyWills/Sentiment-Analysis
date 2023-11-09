from keras.models import Sequential
from keras import layers
import dataPreprocessing

# Hyperparameters (still needs to be adjusted)
max_features = 1000
max_length = 150
epochs = 4

# Preprocessing Steps:
read_data = dataPreprocessing.load_data('data/processed_data.csv')       # load in data
dataPreprocessing.clean_data(read_data)                                  # clean data
dataPreprocessing.summarize(read_data)                                   # summarize data
x_train, x_test, y_train, y_test = dataPreprocessing.split(read_data)    # split data into training and testing sets
x_train, x_test = dataPreprocessing.tokenize(x_train, x_test, max_features, max_length) # tokenize data
print('Preprocessing complete.')

# create empty model
model = Sequential()
# add embedding layer
model.add(layers.Embedding(max_features, 100, input_length=max_length))
# add LSTM layer
model.add(layers.LSTM(100))
# add dense layer
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Model Compiled")

model.fit(x_train, y_train, epochs=epochs, validation_split=0.2)

print("Model Trained")

loss, accuracy = model.evaluate(x_test, y_test)
print('Accuracy: %f' % (accuracy*100))
print('Loss: %f' % (loss*100))

