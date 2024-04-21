import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, f1_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import base64
import os
import tempfile

def get_ml_results():
    # Read the dataset (Make sure the file path is correct)
    df = pd.read_csv('./Cancer.csv')

    # Drop the 'Patient Id' column
    df.drop(['Patient Id'], axis=1, inplace=True)

    # Replace categorical labels with numeric values
    df['Level'].replace('Medium', 'High', inplace=True)
    df['Level'].replace('High', '1', inplace=True)
    df['Level'].replace('Low', '0', inplace=True)
    df['Level'] = pd.to_numeric(df['Level'])

    # Split the data into features and target
    X = df.drop('Level', axis=1)
    y = df['Level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Train a Decision Tree classifier
    tree_ = DecisionTreeClassifier()
    tree_.fit(X_train, y_train)
    y_pred = tree_.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    log_loss_score = log_loss(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Create confusion matrix plot
    plt.figure()
    plt.imshow(np.log(confusion_matrix(y_test, y_pred)), cmap='Blues', interpolation='nearest')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # Save the plot to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        plt.savefig(temp_file)
        temp_file_path = temp_file.name

    # Encode the plot image to base64
    with open(temp_file_path, 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode('utf-8')

    # Remove the temporary plot file
    os.remove(temp_file_path)

    # Return the evaluation metrics and encoded image as a dictionary
    return {
        'accuracy': accuracy,
        'log_loss_score': log_loss_score,
        'f1_score': f1,
        'confusion_matrix_image': encoded_image
    }
