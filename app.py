from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('logistic_regression_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from the form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Create input array for prediction
        input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Make prediction
        prediction = model.predict(input_features)

        # Map prediction to class name
        iris_class = ['Setosa', 'Versicolor', 'Virginica']
        predicted_class = iris_class[prediction[0]]

        return render_template('index.html', prediction=predicted_class)
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)

# Example:
# Number to test predict
# 5.1
# 3.5
# 1.4
# 0.2
# Setosa

#  Iris Versicolor:
# 7.0
# 3.2
# 4.7
# 1.4

# Iris VirgiSnica:
# 6.3
# 3.3
# 6.0
# 2.5
