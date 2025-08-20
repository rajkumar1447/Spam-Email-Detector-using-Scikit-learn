# Import the necessary packages.
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path="data/spam.csv"):
    data = pd.read_csv(path)
    x = data.drop('spam', axis=1)
    y = data['spam']
    return train_test_split(x, y, test_size=0.2, random_state=42)
