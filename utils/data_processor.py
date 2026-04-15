import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

class DataProcessor:
    def __init__(self):
        self.device_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Define all possible categories
        self.all_devices = ['mobile', 'laptop', 'tablet']
        self.all_locations = ['USA', 'Canada', 'UK', 'Germany', 'India', 'Singapore', 
                              'Japan', 'South Korea', 'Australia', 'New Zealand', 
                              'Brazil', 'Russia', 'China', 'France']
    
    def fit_encoders(self, df):
        """Fit encoders on training data"""
        # Fit device encoder with all possible devices
        self.device_encoder.fit(self.all_devices)
        
        # Fit location encoder with all possible locations
        self.location_encoder.fit(self.all_locations)
        
        # First encode the categorical features
        df_encoded = df.copy()
        df_encoded['device_encoded'] = self.device_encoder.transform(df['device'])
        df_encoded['location_encoded'] = self.location_encoder.transform(df['location'])
        
        # Fit scaler on numerical features
        numerical_features = ['login_hour', 'device_encoded', 'location_encoded']
        self.scaler.fit(df_encoded[numerical_features])
        
        self.is_fitted = True
    
    def transform_features(self, df):
        """Transform categorical features to numerical"""
        if not self.is_fitted:
            raise ValueError("Encoders not fitted. Call fit_encoders first.")
        
        df = df.copy()
        
        # Encode categorical features
        df['device_encoded'] = self.device_encoder.transform(df['device'])
        df['location_encoded'] = self.location_encoder.transform(df['location'])
        
        # Select features for scaling
        numerical_features = ['login_hour', 'device_encoded', 'location_encoded']
        
        # Scale numerical features
        df[numerical_features] = self.scaler.transform(df[numerical_features])
        
        return df
    
    def prepare_features(self, df, fit_encoders=False):
        """Prepare features for model training/prediction"""
        if fit_encoders:
            self.fit_encoders(df)
        
        # Transform features
        processed_df = self.transform_features(df)
        
        # Select feature columns
        feature_columns = ['login_hour', 'device_encoded', 'location_encoded']
        X = processed_df[feature_columns]
        
        return X, processed_df
    
    def preprocess_login_attempt(self, user_id, login_hour, device, location):
        """Preprocess a single login attempt for prediction"""
        # Create DataFrame for single login attempt
        login_data = pd.DataFrame({
            'user_id': [user_id],
            'login_hour': [login_hour],
            'device': [device],
            'location': [location]
        })
        
        # Transform features
        if not self.is_fitted:
            raise ValueError("DataProcessor not fitted. Call fit_encoders first.")
        
        X, _ = self.prepare_features(login_data)
        
        return X
    
    def save_encoders(self, model_dir='models'):
        """Save fitted encoders and scaler"""
        if not self.is_fitted:
            raise ValueError("No encoders to save. Fit encoders first.")
        
        os.makedirs(model_dir, exist_ok=True)
        
        with open(f'{model_dir}/device_encoder.pkl', 'wb') as f:
            pickle.dump(self.device_encoder, f)
        
        with open(f'{model_dir}/location_encoder.pkl', 'wb') as f:
            pickle.dump(self.location_encoder, f)
        
        with open(f'{model_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print("Encoders and scaler saved successfully.")
    
    def load_encoders(self, model_dir='models'):
        """Load fitted encoders and scaler"""
        try:
            with open(f'{model_dir}/device_encoder.pkl', 'rb') as f:
                self.device_encoder = pickle.load(f)
            
            with open(f'{model_dir}/location_encoder.pkl', 'rb') as f:
                self.location_encoder = pickle.load(f)
            
            with open(f'{model_dir}/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.is_fitted = True
            print("Encoders and scaler loaded successfully.")
        except FileNotFoundError:
            raise FileNotFoundError("Encoder files not found. Make sure to train the model first.")
    
    def get_feature_names(self):
        """Get feature names for model input"""
        return ['login_hour', 'device_encoded', 'location_encoded']
    
    def inverse_transform_device(self, encoded_device):
        """Convert encoded device back to original label"""
        return self.device_encoder.inverse_transform([encoded_device])[0]
    
    def inverse_transform_location(self, encoded_location):
        """Convert encoded location back to original label"""
        return self.location_encoder.inverse_transform([encoded_location])[0]
