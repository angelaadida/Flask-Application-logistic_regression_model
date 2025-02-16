Project Overview

      This project builds a Logistic Regression Model using Scikit-Learn to classify flowers in the Iris dataset. 
      
      The trained model is saved using Joblib for future predictions.

1. Libraries Used
      joblib: Saves and loads the trained machine learning model.
      sklearn.datasets.load_iris: Loads the Iris dataset for classification.
      sklearn.model_selection.train_test_split: Splits data into training and testing sets.
      sklearn.linear_model.LogisticRegression: Implements Logistic Regression for classification.
      sklearn.metrics.accuracy_score: Measures model accuracy.

2. Loading the Dataset – Iris Dataset
      data = load_iris()
      X = data.data
      y = data.target
      The Iris dataset contains 150 samples of iris flowers.
      Features (X):
      Sepal length
      Sepal width
      Petal length
      Petal width
      Target (y):
      Classifies flowers into three species (Setosa, Versicolor, and Virginica).


3. Splitting Data into Training and Testing Sets

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
      70% training data and 30% testing data.
      random_state=42 ensures consistent results.    


4. Initializing the Logistic Regression Model

    model = LogisticRegression(max_iter=200)
    Logistic Regression is used for classification.
    max_iter=200: Increases the number of iterations to ensure convergence.
    

5. Training the Model

    model.fit(X_train, y_train)
    Trains the logistic regression model using the training data.


6. Evaluating the Model
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    Uses model.predict() to classify test data.
    Measures accuracy using accuracy_score().


7. Saving the Trained Model Using Joblib
    joblib.dump(model, 'logistic_regression_model.pkl')
    Saves the trained model as logistic_regression_model.pkl.
    Joblib is optimized for saving large models efficiently.


8. Loading the Model for Future Predictions
    loaded_model = joblib.load('logistic_regression_model.pkl')
    Loads the saved logistic regression model from disk.


9. Making Predictions (Example Usage)
     import numpy as np
     Uses model.predict() to classify the flower.
     Maps prediction to the Iris species name.


Summary
      Load the Iris dataset (features: Sepal & Petal dimensions, target: flower species).
      Split data into training (70%) and testing (30%).
      Train a Logistic Regression Model using Scikit-Learn.
      Evaluate the model using accuracy score.
      Save the trained model using Joblib.
      Load the saved model for predictions.


Deployment Options
      Convert this into a Flask API to serve predictions via a web interface.
      Deploy on Heroku, AWS, or Flask locally.
