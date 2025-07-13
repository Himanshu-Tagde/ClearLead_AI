import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class LeadScoringModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        
    def preprocess_data(self, df):
        """Clean and preprocess the dataset"""
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Handle missing values
        data = data.fillna('Unknown')
        
        # Create binary target variable (1 if converted, 0 otherwise)
        data['target'] = data['Converted'].astype(int)
        
        # Feature engineering
        features = []
        
        # Numerical features
        numerical_features = ['TotalVisits', 'Total Time Spent on Website', 
                            'Page Views Per Visit', 'Asymmetrique Activity Score', 
                            'Asymmetrique Profile Score']
        
        for col in numerical_features:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
                features.append(col)
        
        # Categorical features to encode
        categorical_features = ['Lead Origin', 'Lead Source', 'Country', 'Specialization', 
                              'What is your current occupation', 'Lead Quality',
                              'Do Not Email', 'Do Not Call']
        
        for col in categorical_features:
            if col in data.columns:
                # Create label encoder for this column
                le = LabelEncoder()
                data[f'{col}_encoded'] = le.fit_transform(data[col].astype(str))
                self.label_encoders[col] = le
                features.append(f'{col}_encoded')
        
        # Binary features
        binary_features = ['Search', 'Magazine', 'Newspaper Article', 'X Education Forums',
                          'Newspaper', 'Digital Advertisement', 'Through Recommendations',
                          'Receive More Updates About Our Courses']
        
        for col in binary_features:
            if col in data.columns:
                data[f'{col}_binary'] = (data[col] == 'Yes').astype(int)
                features.append(f'{col}_binary')
        
        # Activity index features
        activity_features = ['Asymmetrique Activity Index', 'Asymmetrique Profile Index']
        for col in activity_features:
            if col in data.columns:
                # Convert to numerical (High=3, Medium=2, Low=1)
                activity_map = {'01.High': 3, '02.Medium': 2, '03.Low': 1}
                data[f'{col}_num'] = data[col].map(activity_map).fillna(0)
                features.append(f'{col}_num')
        
        self.feature_columns = features
        return data[features + ['target']]
    
    def train(self, dataset_path):
        """Train the lead scoring model"""
        # Load dataset
        df = pd.read_csv(dataset_path)
        print(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns")
        
        # Preprocess data
        processed_data = self.preprocess_data(df)
        
        # Separate features and target
        X = processed_data[self.feature_columns]
        y = processed_data['target']
        
        print(f"Training with {len(X)} samples and {len(self.feature_columns)} features")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return self.model
    
    def predict_score(self, lead_data):
        """Predict lead score for a single lead"""
        # Convert to DataFrame for processing
        df = pd.DataFrame([lead_data])
        
        # Preprocess the data
        processed_data = self.preprocess_single_lead(df)
        
        # Scale features
        X_scaled = self.scaler.transform(processed_data[self.feature_columns])
        
        # Get probability score
        probability = self.model.predict_proba(X_scaled)[0, 1]
        
        # Scale to 0-100
        score = int(probability * 100)
        
        return score
    
    def preprocess_single_lead(self, df):
        """Preprocess a single lead for prediction"""
        data = df.copy()
        data = data.fillna('Unknown')
        
        # Numerical features
        numerical_features = ['TotalVisits', 'Total Time Spent on Website', 
                            'Page Views Per Visit', 'Asymmetrique Activity Score', 
                            'Asymmetrique Profile Score']
        
        for col in numerical_features:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
        
        # Categorical features
        categorical_features = ['Lead Origin', 'Lead Source', 'Country', 'Specialization', 
                              'What is your current occupation', 'Lead Quality',
                              'Do Not Email', 'Do Not Call']
        
        for col in categorical_features:
            if col in data.columns and col in self.label_encoders:
                # Handle unknown categories
                try:
                    data[f'{col}_encoded'] = self.label_encoders[col].transform(data[col].astype(str))
                except ValueError:
                    # If unseen category, assign 0
                    data[f'{col}_encoded'] = 0
        
        # Binary features
        binary_features = ['Search', 'Magazine', 'Newspaper Article', 'X Education Forums',
                          'Newspaper', 'Digital Advertisement', 'Through Recommendations',
                          'Receive More Updates About Our Courses']
        
        for col in binary_features:
            if col in data.columns:
                data[f'{col}_binary'] = (data[col] == 'Yes').astype(int)
        
        # Activity index features
        activity_features = ['Asymmetrique Activity Index', 'Asymmetrique Profile Index']
        for col in activity_features:
            if col in data.columns:
                activity_map = {'01.High': 3, '02.Medium': 2, '03.Low': 1}
                data[f'{col}_num'] = data[col].map(activity_map).fillna(0)
        
        return data
    
    def save_model(self, model_path):
        """Save the trained model and preprocessors"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load the trained model and preprocessors"""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        print(f"Model loaded from {model_path}")

# Training script
if __name__ == "__main__":
    # Initialize the model
    lead_model = LeadScoringModel()
    
    # Train the model (replace with your dataset path)
    dataset_path = "dataset.csv"  # Change this to your actual dataset path
    
    try:
        lead_model.train(dataset_path)
        
        # Save the model
        model_path = "lead_scoring_model.pkl"
        lead_model.save_model(model_path)
        
        print(f"\nTraining completed! Model saved as {model_path}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        
    # Test prediction with sample data
    sample_lead = {
        'TotalVisits': 2,
        'Total Time Spent on Website': 1000,
        'Page Views Per Visit': 2.5,
        'Lead Origin': 'Landing Page Submission',
        'Lead Source': 'Google',
        'Country': 'India',
        'Specialization': 'Business Administration',
        'What is your current occupation': 'Student',
        'Lead Quality': 'Might be',
        'Do Not Email': 'No',
        'Do Not Call': 'No',
        'Asymmetrique Activity Score': 15,
        'Asymmetrique Profile Score': 18,
        'Asymmetrique Activity Index': '02.Medium',
        'Asymmetrique Profile Index': '01.High',
        'Search': 'No',
        'Magazine': 'No',
        'Digital Advertisement': 'No',
        'Through Recommendations': 'No'
    }
    
    try:
        score = lead_model.predict_score(sample_lead)
        print(f"\nSample lead score: {score}/100")
    except Exception as e:
        print(f"Error in prediction: {str(e)}")