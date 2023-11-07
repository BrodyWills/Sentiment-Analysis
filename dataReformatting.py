import pandas as pd

# Reformat the original dataset to a new dataset containing only the columns needed
original_data_path = 'data/training.1600000.processed.noemoticon.csv'
output_data_path = 'data/processed_data.csv'


def main():
    # Read in two needed columns of dataset
    data = pd.read_csv(original_data_path, header=None, encoding='latin-1', usecols=[0, 5], names=['label', 'text'])
    # Shuffle the rows
    data = data.sample(frac=1)
    # Save the new dataset
    data.to_csv(output_data_path, index=False)


if __name__ == '__main__':
    main()
