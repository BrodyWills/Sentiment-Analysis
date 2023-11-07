from keras.models import Sequential
from keras import layers
import dataPreprocessing

# Preprocessing Steps:
read_data = dataPreprocessing.load_data('data/processed_data.csv')
dataPreprocessing.clean_data(read_data)
tokenized_data = dataPreprocessing.tokenize(read_data)
read_data['text'] = tokenized_data
x_train, x_test, y_train, y_test = dataPreprocessing.split(read_data)

# create empty model
model = Sequential()

# TODO: Adjust parameters (just using random default values right now)

# add embedding layer
model.add(layers.Embedding(input_dim=5000, output_dim=50))
# add LSTM layer
model.add(layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2))
# add dense layer
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=1, validation_split=0.2)

loss, accuracy = model.evaluate(x_test, y_test)
print('Accuracy: %f' % (accuracy*100))
print('Loss: %f' % (loss*100))

