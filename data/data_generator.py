import pandas as pd
import numpy as np
from datetime import datetime
import random

class LoginDataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
        # Define normal patterns for users
        self.user_patterns = {
            'user_001': {
                'normal_hours': [9, 10, 11, 14, 15, 16, 17, 18],
                'preferred_devices': ['laptop', 'mobile'],
                'locations': ['USA', 'Canada']
            },
            'user_002': {
                'normal_hours': [8, 9, 10, 11, 13, 14, 15],
                'preferred_devices': ['laptop', 'tablet'],
                'locations': ['UK', 'Germany']
            },
            'user_003': {
                'normal_hours': [19, 20, 21, 22, 23],
                'preferred_devices': ['mobile', 'tablet'],
                'locations': ['India', 'Singapore']
            },
            'user_004': {
                'normal_hours': [7, 8, 9, 17, 18, 19],
                'preferred_devices': ['laptop'],
                'locations': ['Japan', 'South Korea']
            },
            'user_005': {
                'normal_hours': [10, 11, 12, 13, 14, 15],
                'preferred_devices': ['mobile', 'laptop', 'tablet'],
                'locations': ['Australia', 'New Zealand']
            }
        }
        
        # All possible values
        self.all_devices = ['mobile', 'laptop', 'tablet']
        self.all_locations = ['USA', 'Canada', 'UK', 'Germany', 'India', 'Singapore', 
                              'Japan', 'South Korea', 'Australia', 'New Zealand', 
                              'Brazil', 'Russia', 'China', 'France']
    
    def generate_normal_logins(self, num_samples=1000):
        """Generate normal login patterns based on user behavior"""
        data = []
        
        for _ in range(num_samples):
            user_id = random.choice(list(self.user_patterns.keys()))
            pattern = self.user_patterns[user_id]
            
            # Generate normal login
            login_data = {
                'user_id': user_id,
                'login_hour': random.choice(pattern['normal_hours']),
                'device': random.choice(pattern['preferred_devices']),
                'location': random.choice(pattern['locations']),
                'is_anomaly': 0
            }
            data.append(login_data)
        
        return pd.DataFrame(data)
    
    def generate_anomalous_logins(self, num_samples=200):
        """Generate anomalous login patterns"""
        data = []
        
        for _ in range(num_samples):
            user_id = random.choice(list(self.user_patterns.keys()))
            pattern = self.user_patterns[user_id]
            
            # Generate anomalous login with various anomaly types
            anomaly_type = random.choice(['unusual_time', 'new_device', 'unusual_location', 'combined'])
            
            if anomaly_type == 'unusual_time':
                # Login at unusual hours (early morning or late night)
                unusual_hours = [h for h in range(24) if h not in pattern['normal_hours']]
                login_data = {
                    'user_id': user_id,
                    'login_hour': random.choice(unusual_hours),
                    'device': random.choice(pattern['preferred_devices']),
                    'location': random.choice(pattern['locations']),
                    'is_anomaly': 1
                }
            
            elif anomaly_type == 'new_device':
                # Login from a device not normally used
                new_devices = [d for d in self.all_devices if d not in pattern['preferred_devices']]
                if not new_devices:
                    # If all devices are preferred, use a random device but still mark as anomaly
                    device = random.choice(self.all_devices)
                else:
                    device = random.choice(new_devices)
                login_data = {
                    'user_id': user_id,
                    'login_hour': random.choice(pattern['normal_hours']),
                    'device': device,
                    'location': random.choice(pattern['locations']),
                    'is_anomaly': 1
                }
            
            elif anomaly_type == 'unusual_location':
                # Login from an unusual location
                new_locations = [l for l in self.all_locations if l not in pattern['locations']]
                if not new_locations:
                    # If all locations are covered, use a random location but still mark as anomaly
                    location = random.choice(self.all_locations)
                else:
                    location = random.choice(new_locations)
                login_data = {
                    'user_id': user_id,
                    'login_hour': random.choice(pattern['normal_hours']),
                    'device': random.choice(pattern['preferred_devices']),
                    'location': location,
                    'is_anomaly': 1
                }
            
            else:  # combined
                # Multiple anomalies combined
                unusual_hours = [h for h in range(24) if h not in pattern['normal_hours']]
                new_devices = [d for d in self.all_devices if d not in pattern['preferred_devices']]
                new_locations = [l for l in self.all_locations if l not in pattern['locations']]
                
                # Handle empty lists
                if not unusual_hours:
                    unusual_hours = list(range(24))
                if not new_devices:
                    new_devices = self.all_devices
                if not new_locations:
                    new_locations = self.all_locations
                
                login_data = {
                    'user_id': user_id,
                    'login_hour': random.choice(unusual_hours),
                    'device': random.choice(new_devices),
                    'location': random.choice(new_locations),
                    'is_anomaly': 1
                }
            
            data.append(login_data)
        
        return pd.DataFrame(data)
    
    def generate_dataset(self, normal_samples=1000, anomalous_samples=200):
        """Generate complete dataset"""
        normal_data = self.generate_normal_logins(normal_samples)
        anomalous_data = self.generate_anomalous_logins(anomalous_samples)
        
        # Combine and shuffle
        complete_data = pd.concat([normal_data, anomalous_data], ignore_index=True)
        complete_data = complete_data.sample(frac=1).reset_index(drop=True)
        
        return complete_data
    
    def save_dataset(self, df, filename='login_data.csv'):
        """Save dataset to CSV file"""
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")
        print(f"Total samples: {len(df)}")
        print(f"Normal logins: {len(df[df['is_anomaly'] == 0])}")
        print(f"Anomalous logins: {len(df[df['is_anomaly'] == 1])}")

if __name__ == "__main__":
    generator = LoginDataGenerator()
    dataset = generator.generate_dataset(normal_samples=1000, anomalous_samples=200)
    generator.save_dataset(dataset, '../data/login_data.csv')
