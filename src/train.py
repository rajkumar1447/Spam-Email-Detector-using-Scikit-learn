# Import the necessary packages.
from src.data_loader import load_data
from src.model import build_model

def train_model():
    x_train, x_test, y_train, y_test = load_data()
    model = build_model()
    model.fit(x_train, y_train)
    return model, x_test, y_test
