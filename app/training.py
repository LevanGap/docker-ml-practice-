import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
import pickle
import os
import sklearn

print(sklearn.__version__)
# --- 1. Load Data ---
print("1. Loading the Iris dataset...")
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')

print(f"Dataset shape: X={X.shape}, y={y.shape}")
print("First 5 rows of features:\n", X.head())
print("First 5 target labels:\n", y.head())

# --- 2. Split Data ---
print("\n2. Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")

# --- 3. Initialize and Train Model ---
print("\n3. Initializing and training Logistic Regression model...")
model = LogisticRegression(max_iter=200, random_state=42) # Increased max_iter to ensure convergence
model.fit(X_train, y_train)
print("Model training complete!")

# --- 4. Evaluate Model ---
print("\n4. Evaluating the model on the test set...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))


# --- 5. Save the Trained Model to .pkl ---
# Define the filename for your model
model_filename = 'lr_model.pkl'

print(f"\n5. Saving the trained model to '{model_filename}'...")
try:
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model successfully saved to {os.path.abspath(model_filename)}")
except Exception as e:
    print(f"Error saving model: {e}")

# --- 6. Load the Model (Demonstration) ---
print(f"\n6. Demonstrating loading the model from '{model_filename}'...")
try:
    with open(model_filename, 'rb') as file:
        loaded_model = pickle.load(file)
    print("Model successfully loaded!")

    # Test the loaded model
    loaded_model_predictions = loaded_model.predict(X_test)
    loaded_accuracy = accuracy_score(y_test, loaded_model_predictions)
    print(f"Accuracy of the loaded model: {loaded_accuracy:.4f} (should match original accuracy)")

except FileNotFoundError:
    print(f"Error: Model file '{model_filename}' not found.")
except Exception as e:
    print(f"Error loading model: {e}")

print(sklearn.__version__)
