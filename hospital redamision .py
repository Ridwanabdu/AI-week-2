"""
Hospital Readmission Prediction Model Training
This script trains a machine learning model to predict 30-day readmission risk
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

class ReadmissionPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None

    def generate_sample_data(self, n_samples=1000):
        """Generate synthetic patient data for demonstration"""
        print(f"Generating {n_samples} synthetic patient records...")

        data = {
            'age': np.random.randint(18, 95, n_samples),
            'gender': np.random.choice(['male', 'female'], n_samples),
            'admission_type': np.random.choice(['emergency', 'urgent', 'elective'], n_samples, p=[0.4, 0.3, 0.3]),
            'diagnosis': np.random.choice(['diabetes', 'heart_disease', 'respiratory', 'kidney', 'other'], n_samples),
            'num_medications': np.random.randint(0, 20, n_samples),
            'num_procedures': np.random.randint(0, 10, n_samples),
            'length_of_stay': np.random.randint(1, 30, n_samples),
            'num_lab_procedures': np.random.randint(10, 100, n_samples),
            'num_diagnoses': np.random.randint(1, 15, n_samples),
            'prior_admissions': np.random.randint(0, 10, n_samples),
            'systolic_bp': np.random.randint(90, 180, n_samples),
            'diastolic_bp': np.random.randint(60, 120, n_samples),
            'heart_rate': np.random.randint(50, 120, n_samples),
            'blood_sugar': np.random.randint(70, 400, n_samples),
            'has_diabetes': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'has_hypertension': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        }

        df = pd.DataFrame(data)

        # Generate readmission target based on risk factors
        risk_score = (
            (df['age'] > 65).astype(int) * 0.2 +
            (df['prior_admissions'] > 2).astype(int) * 0.3 +
            (df['num_medications'] > 10).astype(int) * 0.2 +
            (df['length_of_stay'] > 7).astype(int) * 0.15 +
            (df['has_diabetes'] == 1).astype(int) * 0.15 +
            np.random.random(n_samples) * 0.3
        )
        df['readmitted'] = (risk_score > 0.5).astype(int)

        return df

    def preprocess_data(self, df, fit=True):
        """Preprocess the data for model training"""
        print("Preprocessing data...")

        # Separate features and target
        if 'readmitted' in df.columns:
          X = df.drop('readmitted', axis=1)
          y = df['readmitted']
        else:
          X = df
          y = None


        # Encode categorical variables
        categorical_cols = ['gender', 'admission_type', 'diagnosis']

        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                # Handle unseen labels during prediction
                if col in self.label_encoders:
                    # Use transform and handle potential errors
                    try:
                        X[col] = self.label_encoders[col].transform(X[col])
                    except ValueError as e:
                        print(f"Warning: Unseen labels in column '{col}' during transformation. Error: {e}")
                        # Option 1: Impute with a placeholder (e.g., -1 or a new category)
                        # X[col] = X[col].apply(lambda x: self.label_encoders[col].transform([x])[0] if x in self.label_encoders[col].classes_ else -1)
                        # Option 2: Convert to string and then transform (if applicable)
                        X[col] = X[col].astype(str)
                        X[col] = self.label_encoders[col].transform(X[col])


                else:
                    print(f"Warning: Label encoder for column '{col}' not found. Skipping transformation.")


        # Scale numerical features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.feature_names = X.columns.tolist()
        else:
            X_scaled = self.scaler.transform(X)

        return X_scaled, y


    def train_model(self, X_train, y_train):
        """Train the readmission prediction model"""
        print("\nTraining Random Forest model...")

        # Use RandomForestClassifier for better interpretability
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"Cross-validation ROC-AUC scores: {cv_scores}")
        print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        print("\nEvaluating model performance...")

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Not Readmitted', 'Readmitted']))

        # ROC-AUC score
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC-AUC Score: {roc_auc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('confusion_matrix.png')
        print("Confusion matrix saved to 'confusion_matrix.png'")

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig('roc_curve.png')
        print("ROC curve saved to 'roc_curve.png'")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))

        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("Feature importance plot saved to 'feature_importance.png'")

        return roc_auc

    def save_model(self, filename='readmission_model.pkl'):
        """Save the trained model and preprocessors"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'timestamp': datetime.now().isoformat()
            }

        joblib.dump(model_data, filename)
        print(f"\nModel saved to '{filename}'")

    def load_model(self, filename='readmission_model.pkl'):
        """Load a trained model"""
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from '{filename}'")
        print(f"Model trained on: {model_data['timestamp']}")

    def predict(self, patient_data):
        """Predict readmission risk for a single patient"""
        # Convert to DataFrame if dict
        if isinstance(patient_data, dict):
            patient_data = pd.DataFrame([patient_data])

        # Ensure the order of columns matches the training data
        if self.feature_names is not None:
            patient_data = patient_data[self.feature_names]
        else:
            print("Warning: Feature names not available. Assuming input data column order matches training data.")


        # Preprocess the data (only features)
        # The preprocess_data method handles dropping 'readmitted' if it exists
        X_scaled, _ = self.preprocess_data(patient_data, fit=False)


        # Predict
        risk_probability = self.model.predict_proba(X_scaled)[0, 1]
        risk_category = 'high' if risk_probability > 0.7 else 'medium' if risk_probability > 0.3 else 'low'

        return {
            'risk_score': float(risk_probability),
            'risk_category': risk_category,
            'readmission_predicted': bool(self.model.predict(X_scaled)[0])
        }

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("Hospital Readmission Prediction Model Training")
    print("=" * 60)

    # Initialize predictor
    predictor = ReadmissionPredictor()

    # Generate sample data (replace with your actual data)
    df = predictor.generate_sample_data(n_samples=2000)
    print(f"\nDataset shape: {df.shape}")
    print(f"Readmission rate: {df['readmitted'].mean():.2%}")

    # Preprocess data
    X, y = predictor.preprocess_data(df, fit=True)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Train model
    predictor.train_model(X_train, y_train)

    # Evaluate model
    roc_auc = predictor.evaluate_model(X_test, y_test)

    # Save model
    predictor.save_model('readmission_model.pkl')

    # Example prediction
    print("\n" + "=" * 60)
    print("Example Prediction:")
    print("=" * 60)
    sample_patient = {
        'age': 72,
        'gender': 'male',
        'admission_type': 'emergency',
        'diagnosis': 'heart_disease',
        'num_medications': 15,
        'num_procedures': 3,
        'length_of_stay': 8,
        'num_lab_procedures': 65,
        'num_diagnoses': 7,
        'prior_admissions': 4,
        'systolic_bp': 145,
        'diastolic_bp': 90,
        'heart_rate': 85,
        'blood_sugar': 180,
        'has_diabetes': 1,
        'has_hypertension': 1,
    }
    result = predictor.predict(sample_patient)
    print(f"\nPatient: {sample_patient['age']}y {sample_patient['gender']}, {sample_patient['diagnosis']}")
    print(f"Risk Score: {result['risk_score']:.2%}")
    print(f"Risk Category: {result['risk_category'].upper()}")
    print(f"Readmission Predicted: {result['readmission_predicted']}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

