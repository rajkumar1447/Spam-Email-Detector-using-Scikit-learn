# Import necessary Libraries.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data set.
data = pd.read_csv("spam.csv")
x = data.drop('spam', axis=1)
y = data['spam']

# Splitting the data set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# train the logistic regression model.
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print(x_test)

# Evaluate the the model using accuracy, confusion matrix, precision, recall and F1 score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Visualize the confusion matrix using Seaborn heatmap
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrx')
plt.show()