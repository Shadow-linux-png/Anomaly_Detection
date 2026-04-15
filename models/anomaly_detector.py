import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
from utils.data_processor import DataProcessor

class AnomalyDetector:
    def __init__(self, contamination=0.1, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.data_processor = DataProcessor()
        self.is_trained = False
        
        # Store training data statistics for explanation
        self.training_stats = {}
    
    def train(self, df):
        """Train the anomaly detection model"""
        print("Training anomaly detection model...")
        
        # Prepare features
        X, processed_df = self.data_processor.prepare_features(df, fit_encoders=True)
        
        # Train model on all data (unsupervised learning)
        self.model.fit(X)
        
        # Calculate training statistics for explanation
        self._calculate_training_stats(processed_df)
        
        # Save data processor
        self.data_processor.save_encoders()
        
        # Save model
        self.save_model()
        
        self.is_trained = True
        print("Model training completed successfully.")
        
        # Evaluate on training data
        self._evaluate_training_data(X, df['is_anomaly'])
    
    def _calculate_training_stats(self, df):
        """Calculate statistics from training data for anomaly explanation"""
        self.training_stats = {
            'login_hour_mean': df['login_hour'].mean(),
            'login_hour_std': df['login_hour'].std(),
            'device_distribution': df['device'].value_counts().to_dict(),
            'location_distribution': df['location'].value_counts().to_dict()
        }
    
    def predict_anomaly(self, user_id, login_hour, device, location):
        """Predict if a login attempt is anomalous"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Preprocess the login attempt
        X = self.data_processor.preprocess_login_attempt(user_id, login_hour, device, location)
        
        # Get anomaly score and prediction
        anomaly_score = self.model.decision_function(X)[0]
        prediction = self.model.predict(X)[0]
        
        # Convert prediction to human-readable format
        # IsolationForest returns -1 for anomalies, 1 for normal
        is_anomaly = prediction == -1
        
        # Generate explanation if anomalous
        explanation = None
        if is_anomaly:
            explanation = self._generate_explanation(user_id, login_hour, device, location)
        
        result = {
            'user_id': user_id,
            'login_hour': login_hour,
            'device': device,
            'location': location,
            'anomaly_score': float(anomaly_score),
            'is_anomaly': is_anomaly,
            'status': 'suspicious' if is_anomaly else 'normal',
            'explanation': explanation
        }
        
        return result
    
    def _generate_explanation(self, user_id, login_hour, device, location):
        """Generate explanation for why a login is considered anomalous"""
        reasons = []
        
        # Check if login hour is unusual
        hour_z_score = abs(login_hour - self.training_stats['login_hour_mean']) / self.training_stats['login_hour_std']
        if hour_z_score > 2:  # More than 2 standard deviations from mean
            if login_hour < 6 or login_hour > 22:
                reasons.append(f"Unusual login time ({login_hour}:00)")
        
        # Check if device is rare
        device_count = self.training_stats['device_distribution'].get(device, 0)
        total_logins = sum(self.training_stats['device_distribution'].values())
        device_percentage = (device_count / total_logins) * 100
        
        if device_percentage < 5:  # Less than 5% of logins use this device
            reasons.append(f"Rare device usage ({device})")
        
        # Check if location is rare
        location_count = self.training_stats['location_distribution'].get(location, 0)
        location_percentage = (location_count / total_logins) * 100
        
        if location_percentage < 5:  # Less than 5% of logins from this location
            reasons.append(f"Unusual location ({location})")
        
        # If no specific reasons found, provide general explanation
        if not reasons:
            reasons.append("Unusual login pattern detected")
        
        return ", ".join(reasons)
    
    def _evaluate_training_data(self, X, true_labels):
        """Evaluate model performance on training data"""
        predictions = self.model.predict(X)
        # Convert -1, 1 to 1, 0 for comparison with true_labels
        predicted_anomalies = (predictions == -1).astype(int)
        
        print("\nModel Evaluation on Training Data:")
        print(f"True anomalies: {sum(true_labels)}")
        print(f"Predicted anomalies: {sum(predicted_anomalies)}")
        
        print("\nClassification Report:")
        print(classification_report(true_labels, predicted_anomalies))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(true_labels, predicted_anomalies))
    
    def save_model(self, model_path='models/anomaly_detector.pkl'):
        """Save the trained model"""
        os.makedirs('models', exist_ok=True)
        
        model_data = {
            'model': self.model,
            'training_stats': self.training_stats,
            'contamination': self.contamination,
            'random_state': self.random_state
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='models/anomaly_detector.pkl'):
        """Load a trained model"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.training_stats = model_data['training_stats']
            self.contamination = model_data['contamination']
            self.random_state = model_data['random_state']
            
            # Load data processor
            self.data_processor.load_encoders()
            
            self.is_trained = True
            print(f"Model loaded from {model_path}")
            
        except FileNotFoundError:
            raise FileNotFoundError("Model file not found. Train the model first.")
    
    def get_anomaly_threshold(self):
        """Get the anomaly score threshold"""
        if not self.is_trained:
            raise ValueError("Model not trained yet.")
        
        # For Isolation Forest, scores below 0 are typically anomalies
        return 0.0
