import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from .models import ModelWithPickle

def ModelKlasifikasi(username, data):
    try:
        name = username
        df = pd.DataFrame(data)
        X1 = df["mackonversi"]
        X2 = df["rssi"]
        y = df["lokasi"]
        X = np.column_stack((X1, X2))
        y = np.array(y)
    
        # =====================SVM n Random Forest Model Start================= #
        # Split data
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        # SVM Model Fungsi
        svm_model_bytes, akurasi_SVM = svm_model(x_train, y_train, x_test, y_test)

        # Random Forest GridSearch Parameter
        rf_model_bytes, akurasi_RF = random_forest_model(x_train, y_train, x_test, y_test)

        # KNN Thariq Model Fungsi
        knn_thariq_bytes, akurasi_KNN_thariq = knn_thariq(x_train, y_train, x_test, y_test)

        #Naive Bayes Model Fungsi
        naive_bayes_model_bytes, akurasi_naive_bayes = naive_bayes_model(x_train, y_train, x_test, y_test)

        # Placeholder untuk 3 model lainnya (belum siap)
        akurasi_knn_farrel = None
        akurasi_neural_network = None

        # Simpan model kedalam database pickle dengan akurasi
        model_instance, created = ModelWithPickle.objects.update_or_create(
            username=name,
            date=pd.Timestamp.now().date(),
            time=pd.Timestamp.now(),
            defaults={'svm_model': svm_model_bytes, 
                      'rf_model': rf_model_bytes,
                      'knn_thariq_model': knn_thariq_bytes,
                      'naive_bayes_model': naive_bayes_model_bytes,
                      
                      'akurasi_svm': akurasi_SVM,
                      'akurasi_rf': akurasi_RF,
                      'akurasi_knn_thariq': akurasi_KNN_thariq,
                      'akurasi_naive_bayes': akurasi_naive_bayes,
                      'akurasi_knn_farrel': akurasi_knn_farrel,
                      'akurasi_neural_network': akurasi_neural_network
                      }
        )

        status = "Updated" if not created else "Created"
        print(f"{status} model for user '{name}' successfully.")
        
        return(akurasi_SVM, akurasi_RF, akurasi_KNN_thariq, akurasi_naive_bayes, status) # Return akurasi Model and status
    
    except Exception as e:
        print(f"Error: {e}")
        return None, None, f"Error: " + str(e)


def svm_model(x_train, y_train, x_test, y_test):
    param_grid = {'C': [0.1, 1, 10, 100],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf'],
                  'max_iter': [1000, 5000, 10000],
                  'class_weight': [None, 'balanced']
                  }
    svc = SVC()
    grid_svc = GridSearchCV(svc, param_grid, refit=True, verbose=3, cv=5, return_train_score=True)
    grid_svc.fit(x_train, y_train)
    y_pred_svm = grid_svc.predict(x_test)
    akurasi_SVM = round(accuracy_score(y_pred_svm, y_test) * 100, 2)
    svm_model_bytes = pickle.dumps(grid_svc.best_estimator_)
    return svm_model_bytes, akurasi_SVM

def random_forest_model(x_train, y_train, x_test, y_test):
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=0)
    grid_rf = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=3, return_train_score=True)
    grid_rf.fit(x_train, y_train)

    y_pred_rf = grid_rf.predict(x_test)
    akurasi_RF = round(accuracy_score(y_pred_rf, y_test) * 100, 2)
    rf_model_bytes = pickle.dumps(grid_rf.best_estimator_)
    return rf_model_bytes, akurasi_RF
    
def knn_thariq(x_train, y_train, x_test, y_test):
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scaler", "passthrough"),
        ("model", KNeighborsClassifier(metric="euclidean"))
    ])

    param_grid = {
        "model__n_neighbors": [2,3,4,5,6,7,8,9,10],
        "model__weights": ["uniform","distance"],
        "scaler": ["passthrough", StandardScaler()]
    }

    grid_knn_thariq = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy",
                        n_jobs=-1, verbose=1)
    grid_knn_thariq.fit(x_train, y_train)

    y_pred_knn_thariq = grid_knn_thariq.predict(x_test)
    akurasi_KNN_thariq = round(accuracy_score(y_pred_knn_thariq, y_test) * 100, 2)
    knn_model_bytes = pickle.dumps(grid_knn_thariq.best_estimator_)
    return knn_model_bytes, akurasi_KNN_thariq


def naive_bayes_model(x_train, y_train, x_test, y_test):
    print("Training Naive Bayes Model...")
    
    # Konversi x_train dan x_test ke DataFrame dengan nama kolom
    x_train_df = pd.DataFrame(x_train, columns=["mackonversi", "rssi"])
    x_test_df = pd.DataFrame(x_test, columns=["mackonversi", "rssi"])
    
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocess = ColumnTransformer([
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", ohe)
        ]), ["mackonversi"]),
        ("num", SimpleImputer(strategy="median"), ["rssi"])
    ], sparse_threshold=0.0)

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", GaussianNB())
    ])

    param_grid = {"model__var_smoothing": [1e-11, 1e-9, 1e-8, 1e-7, 1e-6]}
    print("Performing Grid Search for Naive Bayes...")
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy",
                        n_jobs=-1, verbose=1)
    grid.fit(x_train_df, y_train)
    y_pred_naive_bayes = grid.predict(x_test_df)
    akurasi_naive_bayes = round(accuracy_score(y_pred_naive_bayes, y_test) * 100, 2)
    naive_bayes_model_bytes = pickle.dumps(grid.best_estimator_)
    print("Naive Bayes Model trained successfully.")
    return naive_bayes_model_bytes, akurasi_naive_bayes