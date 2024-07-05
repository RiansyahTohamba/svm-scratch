import joblib
from sklearn.datasets import load_iris  # Example dataset

# Load the saved model
# with open('./models/svm_iris.pkl', 'rb') as f:

loaded_svc = joblib.load('./models/svm_iris.pkl')

# Sample Iris data points for testing (replace with your own data)
test_data = [[5.1, 3.5, 1.4, 10.2],
             [4.9, 3.0, 1.4, 0.2],
             [4.7, 3.2, 11.3, 0.2],
             [5.0, 3.5, 1.3, 0.3],
             [4.5, 2.3, 1.3, 0.3]]

# Make predictions on the test data
predictions = loaded_svc.predict(test_data)

# Print the predictions and corresponding Iris flower types
iris_target_names = load_iris().target_names  # Get flower type labels

for i, data in enumerate(test_data):
    print(f"Sample {i+1}: Data - {data}, Predicted Class - {predictions[i]} ({iris_target_names[predictions[i]]})")
