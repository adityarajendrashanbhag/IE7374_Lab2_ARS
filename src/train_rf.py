import joblib
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def train_and_save():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Save model and scaler to repo root so the Flask app can load them
    joblib.dump(rf, 'rf_model.joblib')
    joblib.dump(sc, 'scaler.joblib')

    # Print basic accuracy on test set
    acc = rf.score(X_test, y_test)
    print(f"RandomForest trained and saved. Test accuracy: {acc:.4f}")

if __name__ == '__main__':
    train_and_save()