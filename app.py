#!/usr/bin/env python3
"""
Streamlit-based AI Login Anomaly Detection System
Integrated with FastAPI backend functionality for complete solution
"""

import streamlit as st
import sys
import os
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import json
import requests
from typing import Optional, Dict, Any

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.anomaly_detector import AnomalyDetector
from data.data_generator import LoginDataGenerator

# Database setup
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'login_history.db')

def init_database():
    """Initialize SQLite database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS login_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                login_hour INTEGER NOT NULL,
                device TEXT NOT NULL,
                location TEXT NOT NULL,
                anomaly_score REAL NOT NULL,
                is_anomaly BOOLEAN NOT NULL,
                status TEXT NOT NULL,
                explanation TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Database initialization failed: {e}")
        return False

def get_login_history(limit=50):
    """Get login history from SQLite"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_id, login_hour, device, location, anomaly_score, is_anomaly, status, explanation, timestamp
            FROM login_attempts
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        columns = [column[0] for column in cursor.description]
        history = []
        for row in cursor.fetchall():
            history.append(dict(zip(columns, row)))
        
        conn.close()
        return history
    except Exception as e:
        print(f"Error getting history: {e}")
        return []

def get_login_stats():
    """Get login statistics from SQLite"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_logins,
                SUM(CASE WHEN status = 'suspicious' THEN 1 ELSE 0 END) as anomalous_logins,
                SUM(CASE WHEN status = 'normal' THEN 1 ELSE 0 END) as normal_logins
            FROM login_attempts
        ''')
        
        stats = cursor.fetchone()
        
        total_logins = stats[0] if stats[0] is not None else 0
        anomalous_logins = stats[1] if stats[1] is not None else 0
        normal_logins = stats[2] if stats[2] is not None else 0
        
        anomaly_rate = (anomalous_logins / total_logins * 100) if total_logins > 0 else 0
        
        conn.close()
        
        return {
            "total_logins": total_logins,
            "normal_logins": normal_logins,
            "anomalous_logins": anomalous_logins,
            "anomaly_rate_percentage": round(anomaly_rate, 2)
        }
    except Exception as e:
        print(f"Error getting stats: {e}")
        return {"total_logins": 0, "normal_logins": 0, "anomalous_logins": 0, "anomaly_rate_percentage": 0}

def save_login_attempt(login_data):
    """Save login attempt to SQLite"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO login_attempts 
            (user_id, login_hour, device, location, anomaly_score, is_anomaly, status, explanation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            login_data['user_id'],
            login_data['login_hour'],
            login_data['device'],
            login_data['location'],
            login_data['anomaly_score'],
            login_data['is_anomaly'],
            login_data['status'],
            login_data.get('explanation')
        ))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving login: {e}")
        return False

def detect_anomaly(user_id, login_hour, device, location):
    """Detect anomaly for given login data"""
    # Use global detector instance
    global detector
    
    if not detector.is_trained:
        return None
    
    try:
        result = detector.predict_anomaly(user_id, login_hour, device, location)
        
        # Save to database
        save_login_attempt(result)
        
        return result
    except Exception as e:
        print(f"Detection failed: {e}")
        return None

def train_model_wrapper(normal_samples=1000, anomalous_samples=200):
    """Wrapper function for training model"""
    try:
        generator = LoginDataGenerator()
        training_data = generator.generate_dataset(normal_samples, anomalous_samples)
        detector.train(training_data)
        return True, f"Model trained successfully with {len(training_data)} samples"
    except Exception as e:
        return False, f"Training failed: {str(e)}"

def get_model_status():
    """Get current model status"""
    return {
        "is_trained": detector.is_trained,
        "status": "ready" if detector.is_trained else "not_trained",
        "message": "Model is ready for anomaly detection" if detector.is_trained else "Model needs to be trained first"
    }

def simulate_random_login():
    """Simulate a random login attempt"""
    import random
    
    users = ['user_001', 'user_002', 'user_003', 'user_004', 'user_005']
    devices = ['mobile', 'laptop', 'tablet']
    locations = ['USA', 'Canada', 'UK', 'Germany', 'India', 'Singapore', 
                 'Japan', 'South Korea', 'Australia', 'New Zealand']
    
    random_login = {
        'user_id': random.choice(users),
        'login_hour': random.randint(0, 23),
        'device': random.choice(devices),
        'location': random.choice(locations)
    }
    
    return detect_anomaly(random_login['user_id'], random_login['login_hour'], 
                         random_login['device'], random_login['location'])

# Initialize database and detector globally
init_database()
detector = AnomalyDetector()

# Try to load existing model
try:
    detector.load_model()
    print("✅ Model loaded successfully")
except FileNotFoundError:
    print("⚠️ No pre-trained model found - will need to train")

# Streamlit UI Configuration
st.set_page_config(
    page_title="AI Login Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #e1e5e9;
    }
    .history-item {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .history-item.anomalous {
        border-left-color: #dc3545;
    }
    .status-normal {
        color: #28a745;
        font-weight: bold;
    }
    .status-anomalous {
        color: #dc3545;
        font-weight: bold;
    }
    .stButton > button {
        width: 100%;
        margin-top: 0.5rem;
    }
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #28a745;
    }
    .result-card.anomalous {
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Main Header
st.markdown("""
<div class="main-header">
    <h1>🔍 AI Login Anomaly Detection System</h1>
    <p>Real-time detection of suspicious login attempts using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'show_api_docs' not in st.session_state:
    st.session_state.show_api_docs = False

# Sidebar for controls
with st.sidebar:
    st.header("🎛 System Controls")
    
    # Model Status
    st.subheader("📊 Model Status")
    model_status = get_model_status()
    
    if model_status['is_trained']:
        st.success("✅ Model is ready for detection")
    else:
        st.warning("⚠️ Model needs training")
        
    if st.button("🤖 Train Model", type="primary"):
        with st.spinner("Training model... This may take a moment."):
            success, message = train_model_wrapper()
            if success:
                st.success(f"🎉 {message}")
                time.sleep(2)
                st.rerun()
            else:
                st.error(f"❌ {message}")
    
    st.divider()
    
    # Anomaly Detection Form
    st.subheader("🧪 Test Login Detection")
    
    # User selection
    user_id = st.selectbox(
        "👤 User ID",
        ['user_001', 'user_002', 'user_003', 'user_004', 'user_005'],
        index=0
    )
    
    # Login inputs
    col1, col2 = st.columns(2)
    with col1:
        login_hour = st.slider("🕐 Login Hour", 0, 23, 10)
        device = st.selectbox("💻 Device", ['mobile', 'laptop', 'tablet'])
    
    with col2:
        location = st.selectbox("🌍 Location", [
            'USA', 'Canada', 'UK', 'Germany', 'India', 'Singapore', 
            'Japan', 'South Korea', 'Australia', 'New Zealand', 
            'Brazil', 'Russia', 'China', 'France'
        ])
    
    # Detection button
    if st.button("🔍 Detect Anomaly", type="primary"):
        if not model_status['is_trained']:
            st.error("❌ Please train the model first!")
        else:
            with st.spinner("Analyzing login..."):
                result = detect_anomaly(user_id, login_hour, device, location)
                if result:
                    st.session_state.last_result = result
                    st.rerun()
                else:
                    st.error("❌ Detection failed")
    
    st.divider()
    
    # Simulation Controls
    st.subheader("🎲 Simulation")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎲 Random Login", type="secondary"):
            if not model_status['is_trained']:
                st.error("❌ Please train the model first!")
            else:
                with st.spinner("Generating random login..."):
                    result = simulate_random_login()
                    if result:
                        st.session_state.last_result = result
                        st.rerun()
                    else:
                        st.error("❌ Simulation failed")
    
    with col2:
        if st.button("🗑️ Clear History", type="secondary"):
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM login_attempts")
                conn.commit()
                conn.close()
                st.success("🗑️ History cleared!")
                st.session_state.last_result = None
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to clear history: {e}")
    
    st.divider()
    
    # API Info
    st.subheader("🔌 API Integration")
    if st.button("📚 API Documentation"):
        st.session_state.show_api_docs = not st.session_state.show_api_docs
        st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Latest Result
    st.header("🎯 Latest Detection Result")
    
    if st.session_state.last_result:
        result = st.session_state.last_result
        
        # Status styling
        if result['is_anomaly']:
            st.markdown(f"""
            <div class="result-card anomalous">
                <h2 style="color: #dc3545;">🚨 SUSPICIOUS LOGIN DETECTED</h2>
                <div style="font-size: 1.2rem; margin: 1rem 0;">
                    <strong>Anomaly Score:</strong> {result['anomaly_score']:.3f}<br>
                    <strong>Explanation:</strong> {result.get('explanation', 'No explanation available')}
                </div>
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                    <strong>Login Details:</strong><br>
                    User: {result['user_id']} | Time: {result['login_hour']}:00<br>
                    Device: {result['device']} | Location: {result['location']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card">
                <h2 style="color: #28a745;">✅ NORMAL LOGIN</h2>
                <div style="font-size: 1.2rem; margin: 1rem 0;">
                    <strong>Anomaly Score:</strong> {result['anomaly_score']:.3f}<br>
                    <strong>Status:</strong> No suspicious activity detected
                </div>
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                    <strong>Login Details:</strong><br>
                    User: {result['user_id']} | Time: {result['login_hour']}:00<br>
                    Device: {result['device']} | Location: {result['location']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("👆 Perform a login detection to see results here")
    
    # Statistics
    st.header("📊 System Statistics")
    
    stats = get_login_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📈 Total Logins", stats['total_logins'])
    
    with col2:
        st.metric("✅ Normal Logins", stats['normal_logins'])
    
    with col3:
        st.metric("🚨 Suspicious Logins", stats['anomalous_logins'])
    
    with col4:
        st.metric("📊 Anomaly Rate", f"{stats['anomaly_rate_percentage']}%")
    
    # Charts
    if stats['total_logins'] > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Login Distribution")
            fig = go.Figure(data=[go.Pie(
                labels=['Normal Logins', 'Suspicious Logins'],
                values=[stats['normal_logins'], stats['anomalous_logins']],
                hole=0.3,
                marker_colors=['#28a745', '#dc3545']
            )])
            
            fig.update_layout(
                title="Login Distribution",
                font=dict(size=14),
                showlegend=True,
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Anomaly Metrics")
            fig_metrics = go.Figure()
            
            fig_metrics.add_trace(go.Bar(
                x=['Total', 'Normal', 'Suspicious'],
                y=[stats['total_logins'], stats['normal_logins'], stats['anomalous_logins']],
                marker_color=['#6c757d', '#28a745', '#dc3545']
            ))
            
            fig_metrics.update_layout(
                title="Login Counts",
                yaxis_title="Count",
                height=300
            )
            
            st.plotly_chart(fig_metrics, use_container_width=True)
    else:
        st.info("📊 No data available for visualization")

with col2:
    # Recent History
    st.header("📜 Recent Login History")
    
    history = get_login_history(15)
    
    if history:
        for item in history:
            status_class = "status-anomalous" if item['status'] == 'suspicious' else "status-normal"
            history_class = "history-item anomalous" if item['status'] == 'suspicious' else "history-item"
            
            st.markdown(f"""
            <div class="{history_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span class="{status_class}">{item['status'].upper()}</span>
                        <span style="margin-left: 10px; color: #666;">Score: {item['anomaly_score']:.3f}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #666;">
                        {datetime.fromisoformat(item['timestamp']).strftime('%m/%d %H:%M')}
                    </div>
                </div>
                <div style="margin-top: 5px; font-size: 0.8rem;">
                    User: {item['user_id']} | Time: {item['login_hour']}:00<br>
                    Device: {item['device']} | Location: {item['location']}
                </div>
                {f"<div style='margin-top: 5px; font-style: italic; color: #666; font-size: 0.8rem;'>💡 {item['explanation']}</div>" if item['explanation'] else ""}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📜 No login history available")
    
    # Auto-refresh
    if st.button("🔄 Refresh Data"):
        st.rerun()

# API Documentation (collapsible)
if st.session_state.show_api_docs:
    st.divider()
    st.header("📚 API Documentation")
    
    st.markdown("""
    ### REST API Endpoints
    
    The system also provides a REST API for integration with external applications:
    
    **Base URL:** `http://localhost:8000` (when running FastAPI server)
    
    #### Main Endpoints:
    
    1. **Health Check**
       - `GET /health`
       - Returns system health status
    
    2. **Model Status**
       - `GET /model/status`
       - Returns current model training status
    
    3. **Train Model**
       - `POST /model/train?normal_samples=1000&anomalous_samples=200`
       - Trains the anomaly detection model
    
    4. **Detect Anomaly**
       - `POST /detect`
       - Body: `{"user_id": "user_001", "login_hour": 10, "device": "mobile", "location": "USA"}`
       - Returns anomaly detection result
    
    5. **Simulate Login**
       - `POST /simulate/login`
       - Generates and tests a random login
    
    6. **Get History**
       - `GET /history?limit=50`
       - Returns recent login history
    
    7. **Get Statistics**
       - `GET /history/stats`
       - Returns login statistics
    
    #### Web Interface:
    - `GET /` - Full HTML web interface with real-time dashboard
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🤖 Built with Streamlit | 💾 SQLite Database | 🧠 Machine Learning</p>
    <p>📚 <strong>Features:</strong> Real-time Detection | History Tracking | Statistics | Visualization | API Integration</p>
</div>
""", unsafe_allow_html=True)
