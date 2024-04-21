import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, f1_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import base64
import os
import tempfile
from flask import Flask, render_template

app = Flask(__name__, template_folder='.')


@app.route('/')
def index():
    return render_template('predict.html')

@app.route('/get_results', methods=['POST'])
def get_results():
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
    print(accuracy, log_loss_score, f1 )
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

    # Generate HTML code to embed the results in the webpage
    html_code = f'''
    <p>Accuracy score: {accuracy * 100:.2f}%</p>
    <p>Log loss: {log_loss_score:.4f}</p>
    <p>F1 score: {f1:.4f}</p>
    <p>Confusion Matrix</p>
    <img src="data:image/png;base64,{encoded_image}" alt="Confusion Matrix">
    '''

    return html_code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080,debug=True)
