# import sklearn
from sklearn import svm
from sklearn.datasets import load_iris  # Example dataset
import joblib
# from sklearn.externals import joblib  # Import joblib for model persistence

# Load the Iris dataset (or use your own data)
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels

# Create and train the SVM classifier
svc = svm.SVC(kernel='linear')  # Customize kernel as needed
svc.fit(X, y)

# print(f"scikit-learn version: {sklearn.__version__}")
# Save the trained model using joblib
joblib.dump(svc, 'svm_iris.pkl')  # Replace 'svm_iris.pkl' with your desired filename

print("SVM classifier saved as svm_iris.pkl")
