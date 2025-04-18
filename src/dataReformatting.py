import pandas as pd

# Reformat the original dataset to a new dataset containing only the columns needed
original_data_path = '../data/Tweets.csv'
output_data_path = '../data/processed_data.csv'


def main():
    # Read in two needed columns of dataset
    data = pd.read_csv(original_data_path, usecols=[1, 10])
    # Rename columns
    data.columns = ['sentiment', 'text']
    # Remove neutral tweets
    data = data[data.sentiment != 'neutral']
    # Convert sentiment to binary
    data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})
    # Save the new dataset
    data.to_csv(output_data_path, index=False)


if __name__ == '__main__':
    main()
