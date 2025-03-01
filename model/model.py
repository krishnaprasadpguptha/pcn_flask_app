import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
#%%
def load_dataset(file_path):
    """Load dataset from CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: Dataset not found at {file_path}")
    df = pd.read_csv(file_path)
    df = pd.read_csv(file_path)
    df = df.rename(columns = {'Vehicle Category': 'VehicleType', 'Contravention Code': 'ContraventionCode', 'Has Appeal': 'AppealOutcome'})

    return df

def preprocess_data(df):
    """Rename columns, handle missing data, and encode categorical variables."""
    column_mapping = {
        "Vehicle Type": "VehicleType",
        "Contravention Code": "ContraventionCode",
        "Location": "Location",
        "Appeal Outcome": "AppealOutcome"
    }
    df.rename(columns=column_mapping, inplace=True)
    
    required_columns = ["VehicleType", "ContraventionCode", "Location", "AppealOutcome"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Error: Missing required columns in dataset: {missing_columns}")
    
    df.dropna(inplace=True)
    
    label_encoders = {}
    for col in ["VehicleType", "ContraventionCode", "Location"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    df["AppealOutcome"] = df["AppealOutcome"].apply(lambda x: 1 if x.lower() == "appealed" else 0)
    
    return df, label_encoders

def feature_selection(df):
    """Select the most relevant features for the model."""
    X = df[["VehicleType", "ContraventionCode", "Location"]]
    y = df["AppealOutcome"].apply(lambda x: 1 if x == "Yes" else 0)  # Convert target to binary
    return X, y

def train_model(X, y):
    """Train a machine learning model using a random forest classifier."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, scaler, X_test, y_test

def validate_model(model, X_test, y_test):
    """Evaluate the trained model using accuracy and classification report."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)

def save_artifacts(model, scaler, label_encoders):
    """Save the trained model, scaler, and label encoders."""
    with open("model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
    with open("scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)
    with open("label_encoders.pkl", "wb") as encoder_file:
        pickle.dump(label_encoders, encoder_file)
def add_model_predictions(model, X, df, file_path):
    df = load_dataset(file_path)
    df, label_encoders = preprocess_data(df)
    df = df.drop(columns = ['AppealOutcome'])
    df['predicted Apeal'] = model.predict(X)
    df.to_csv(r"C:\\Users\\Krishnaprasad\\Desktop\\Assignment\\Assignment_3\\pcn_flask_app\data\Parking_Services_Penalty_Charge_Notices_2019-20_20250205_predicted.csv")
    return df
#%%
def main():
    """Execute the end-to-end ML pipeline."""
    file_path = os.path.join(os.getcwd(), "data", "Parking_Services_Penalty_Charge_Notices_2019-20_20250205.csv")
    df = load_dataset(file_path)
    df, label_encoders = preprocess_data(df)
    X, y = feature_selection(df)
    
    model, scaler, X_test, y_test = train_model(X, y)
    validate_model(model, X_test, y_test)
    save_artifacts(model, scaler, label_encoders)
    add_model_predictions(model, X, df, file_path)
    print("✅ Model training, validation, and saving completed successfully!")

if __name__ == "__main__":
    main()
