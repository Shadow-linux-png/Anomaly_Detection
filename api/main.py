from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import sys
import os
import sqlite3
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.anomaly_detector import AnomalyDetector
from data.data_generator import LoginDataGenerator

app = FastAPI(
    title="AI-based Login Anomaly Detection System",
    description="API for detecting anomalous login attempts using machine learning",
    version="1.0.0"
)

# Initialize model
detector = AnomalyDetector()

# Pydantic models for request/response
class LoginRequest(BaseModel):
    user_id: str
    login_hour: int
    device: str
    location: str

class LoginResponse(BaseModel):
    user_id: str
    login_hour: int
    device: str
    location: str
    anomaly_score: float
    is_anomaly: bool
    status: str
    explanation: Optional[str] = None

class ModelStatus(BaseModel):
    status: str
    is_trained: bool
    message: str

class TrainingResponse(BaseModel):
    message: str
    status: str
    samples_trained: int

# SQLite database setup
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'login_history.db')

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
        print("SQLite database initialized successfully")
        return True
    except Exception as e:
        print(f"Database initialization failed: {e}")
        return False

# Initialize database on startup
init_database()

def save_login_attempt(login_data):
    """Save login attempt to SQLite"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Insert login attempt
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
        print(f"Error saving to SQLite: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    try:
        detector.load_model()
        print("Model loaded successfully on startup")
    except FileNotFoundError:
        print("No pre-trained model found. Please train the model first.")

@app.get("/", response_class=HTMLResponse, tags=["Web Interface"])
async def web_interface():
    """Serve the web interface"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Login Anomaly Detection</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .card h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.5rem;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #555;
        }
        
        .form-group input, .form-group select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 1rem;
            transition: border-color 0.3s ease;
        }
        
        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn-secondary {
            background: #6c757d;
        }
        
        .btn-success {
            background: #28a745;
        }
        
        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            font-weight: 600;
        }
        
        .result.normal {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .result.suspicious {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .stat-item {
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .stat-number {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            color: #666;
            font-size: 0.9rem;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }
        
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        
        @media (max-width: 768px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
            
            .stats {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .header h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>AI Login Anomaly Detection</h1>
            <p>Real-time detection of suspicious login attempts using Machine Learning</p>
        </div>
        
        <div class="dashboard">
            <div class="card">
                <h2>Test Login Detection</h2>
                <form id="detectionForm">
                    <div class="form-group">
                        <label for="userId">User ID:</label>
                        <select id="userId" required>
                            <option value="user_001">User 001</option>
                            <option value="user_002">User 002</option>
                            <option value="user_003">User 003</option>
                            <option value="user_004">User 004</option>
                            <option value="user_005">User 005</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="loginHour">Login Hour (0-23):</label>
                        <input type="number" id="loginHour" min="0" max="23" value="10" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="device">Device:</label>
                        <select id="device" required>
                            <option value="mobile">Mobile</option>
                            <option value="laptop">Laptop</option>
                            <option value="tablet">Tablet</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="location">Location:</label>
                        <select id="location" required>
                            <option value="USA">USA</option>
                            <option value="Canada">Canada</option>
                            <option value="UK">UK</option>
                            <option value="Germany">Germany</option>
                            <option value="India">India</option>
                            <option value="Singapore">Singapore</option>
                            <option value="Japan">Japan</option>
                            <option value="South Korea">South Korea</option>
                            <option value="Australia">Australia</option>
                            <option value="New Zealand">New Zealand</option>
                            <option value="Brazil">Brazil</option>
                            <option value="Russia">Russia</option>
                            <option value="China">China</option>
                            <option value="France">France</option>
                        </select>
                    </div>
                    
                    <button type="submit" class="btn">Detect Anomaly</button>
                </form>
                
                <div id="detectionResult"></div>
            </div>
            
            <div class="card">
                <h2>System Statistics</h2>
                <div class="stats">
                    <div class="stat-item">
                        <div class="stat-number" id="totalLogins">0</div>
                        <div class="stat-label">Total Logins</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number" id="normalLogins">0</div>
                        <div class="stat-label">Normal</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number" id="suspiciousLogins">0</div>
                        <div class="stat-label">Suspicious</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number" id="anomalyRate">0%</div>
                        <div class="stat-label">Anomaly Rate</div>
                    </div>
                </div>
                
                <button class="btn btn-secondary" onclick="simulateLogin()">Simulate Random Login</button>
                <button class="btn btn-success" onclick="trainModel()" style="margin-top: 10px;">Train Model</button>
                
                <div id="systemStatus" style="margin-top: 15px;"></div>
            </div>
        </div>
        
        <div class="card">
            <h2>📜 Recent Login History</h2>
            <div id="loginHistory"><div class="loading">Loading history...</div></div>
            <button class="btn btn-secondary" onclick="refreshHistory()" style="margin-top: 15px;">🔄 Refresh History</button>
        </div>
        </div>
    </div>

    <script>
        // API base URL
        const API_BASE = '';
        
        // Initialize page
        document.addEventListener('DOMContentLoaded', function() {
            checkModelStatus();
            loadStats();
            loadHistory();
            
            // Setup form submission
            document.getElementById('detectionForm').addEventListener('submit', detectAnomaly);
        });
        
        // Check model status
        async function checkModelStatus() {
            try {
                const response = await fetch(`${API_BASE}/model/status`);
                const data = await response.json();
                
                const statusDiv = document.getElementById('systemStatus');
                if (data.is_trained) {
                    statusDiv.innerHTML = '<div style="background: #d4edda; color: #155724; padding: 10px; border-radius: 8px;">Model is ready for detection</div>';
                } else {
                    statusDiv.innerHTML = '<div style="background: #fff3cd; color: #856404; padding: 10px; border-radius: 8px;">Model needs training</div>';
                }
            } catch (error) {
                console.error('Error checking model status:', error);
            }
        }
        
        // Detect anomaly
        async function detectAnomaly(event) {
            event.preventDefault();
            
            const formData = {
                user_id: document.getElementById('userId').value,
                login_hour: parseInt(document.getElementById('loginHour').value),
                device: document.getElementById('device').value,
                location: document.getElementById('location').value
            };
            
            const resultDiv = document.getElementById('detectionResult');
            resultDiv.innerHTML = '<div class="loading">Analyzing login...</div>';
            
            try {
                const response = await fetch(`${API_BASE}/detect`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(formData)
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    const resultClass = data.is_anomaly ? 'suspicious' : 'normal';
                    const statusIcon = data.is_anomaly ? 'SUSPICIOUS' : 'NORMAL';
                    
                    resultDiv.innerHTML = `
                        <div class="result ${resultClass}">
                            <div><strong>Status:</strong> ${statusIcon}</div>
                            <div><strong>Anomaly Score:</strong> ${data.anomaly_score.toFixed(3)}</div>
                            ${data.explanation ? `<div><strong>Explanation:</strong> ${data.explanation}</div>` : ''}
                            <div style="margin-top: 10px; font-size: 0.9rem; opacity: 0.8;">
                                User: ${data.user_id} | Time: ${data.login_hour}:00 | Device: ${data.device} | Location: ${data.location}
                            </div>
                        </div>
                    `;
                    
                    // Refresh stats and history after detection
                    loadStats();
                    loadHistory();
                } else {
                    resultDiv.innerHTML = `<div class="error">Error: ${data.detail}</div>`;
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="error">Network error: ${error.message}</div>`;
            }
        }
        
        // Simulate random login
        async function simulateLogin() {
            const resultDiv = document.getElementById('detectionResult');
            resultDiv.innerHTML = '<div class="loading">Generating random login...</div>';
            
            try {
                const response = await fetch(`${API_BASE}/simulate/login`, {
                    method: 'POST'
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    const resultClass = data.is_anomaly ? 'suspicious' : 'normal';
                    const statusIcon = data.is_anomaly ? 'SUSPICIOUS' : 'NORMAL';
                    
                    resultDiv.innerHTML = `
                        <div class="result ${resultClass}">
                            <div><strong>Status:</strong> ${statusIcon}</div>
                            <div><strong>Anomaly Score:</strong> ${data.anomaly_score.toFixed(3)}</div>
                            ${data.explanation ? `<div><strong>Explanation:</strong> ${data.explanation}</div>` : ''}
                            <div style="margin-top: 10px; font-size: 0.9rem; opacity: 0.8;">
                                User: ${data.user_id} | Time: ${data.login_hour}:00 | Device: ${data.device} | Location: ${data.location}
                            </div>
                        </div>
                    `;
                    
                    // Update form with simulated values
                    document.getElementById('userId').value = data.user_id;
                    document.getElementById('loginHour').value = data.login_hour;
                    document.getElementById('device').value = data.device;
                    document.getElementById('location').value = data.location;
                    
                    // Refresh stats and history after simulation
                    loadStats();
                    loadHistory();
                } else {
                    resultDiv.innerHTML = `<div class="error">Error: ${data.detail}</div>`;
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="error">Network error: ${error.message}</div>`;
            }
        }
        
        // Load statistics
        async function loadStats() {
            try {
                const response = await fetch(`${API_BASE}/history/stats`);
                const data = await response.json();
                
                if (response.ok && data.total_logins > 0) {
                    document.getElementById('totalLogins').textContent = data.total_logins;
                    document.getElementById('normalLogins').textContent = data.normal_logins;
                    document.getElementById('suspiciousLogins').textContent = data.anomalous_logins;
                    document.getElementById('anomalyRate').textContent = data.anomaly_rate_percentage + '%';
                } else {
                    document.getElementById('totalLogins').textContent = '0';
                    document.getElementById('normalLogins').textContent = '0';
                    document.getElementById('suspiciousLogins').textContent = '0';
                    document.getElementById('anomalyRate').textContent = '0%';
                }
            } catch (error) {
                console.error('Error loading stats:', error);
            }
        }
        
        // Load history
        async function loadHistory() {
            const historyDiv = document.getElementById('loginHistory');
            historyDiv.innerHTML = '<div class="loading">Loading history...</div>';
            
            try {
                const response = await fetch(`${API_BASE}/history`);
                const data = await response.json();
                
                if (response.ok && data.history && data.history.length > 0) {
                    let historyHTML = '';
                    data.history.forEach(item => {
                        const statusClass = item.is_anomaly ? 'suspicious' : 'normal';
                        const statusIcon = item.is_anomaly ? '🚨' : '✅';
                        const timestamp = new Date(item.timestamp).toLocaleString();
                        
                        historyHTML += `
                            <div style="background: #f8f9fa; padding: 10px; margin-bottom: 10px; border-radius: 8px; border-left: 4px solid ${item.is_anomaly ? '#dc3545' : '#28a745'};">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <strong>${statusIcon} ${item.status.toUpperCase()}</strong>
                                        <span style="margin-left: 10px; color: #666;">Score: ${item.anomaly_score.toFixed(3)}</span>
                                    </div>
                                    <div style="font-size: 0.9rem; color: #666;">${timestamp}</div>
                                </div>
                                <div style="margin-top: 5px; font-size: 0.9rem;">
                                    User: ${item.user_id} | Time: ${item.login_hour}:00 | Device: ${item.device} | Location: ${item.location}
                                </div>
                                ${item.explanation ? `<div style="margin-top: 5px; font-style: italic; color: #666;">💡 ${item.explanation}</div>` : ''}
                            </div>
                        `;
                    });
                    
                    historyDiv.innerHTML = historyHTML;
                } else {
                    historyDiv.innerHTML = '<div style="color: #666; text-align: center; padding: 20px;">No login history available</div>';
                }
            } catch (error) {
                historyDiv.innerHTML = '<div style="color: #dc3545; text-align: center; padding: 20px;">Error loading login history</div>';
                console.error('Error loading history:', error);
            }
        }
        
        // Refresh history
        async function refreshHistory() {
            await loadStats();
            await loadHistory();
        }
        
        // Train model
        async function trainModel() {
            const resultDiv = document.getElementById('systemStatus');
            resultDiv.innerHTML = '<div class="loading">Training model... This may take a moment.</div>';
            
            try {
                const response = await fetch(`${API_BASE}/model/train?normal_samples=1000&anomalous_samples=200`, {
                    method: 'POST'
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    resultDiv.innerHTML = `<div style="background: #d4edda; color: #155724; padding: 10px; border-radius: 8px;">${data.message} (${data.samples_trained} samples)</div>`;
                    
                    // Refresh status and stats
                    checkModelStatus();
                    loadStats();
                } else {
                    resultDiv.innerHTML = `<div class="error">Training failed: ${data.detail}</div>`;
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="error">Network error: ${error.message}</div>`;
            }
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "login-anomaly-detection"}

@app.get("/model/status", response_model=ModelStatus, tags=["Model"])
async def get_model_status():
    """Get model training status"""
    return ModelStatus(
        status="ready" if detector.is_trained else "not_trained",
        is_trained=detector.is_trained,
        message="Model is ready for anomaly detection" if detector.is_trained else "Model needs to be trained first"
    )

@app.post("/model/train", response_model=TrainingResponse, tags=["Model"])
async def train_model(normal_samples: int = 1000, anomalous_samples: int = 200):
    """Train the anomaly detection model"""
    try:
        # Generate training data
        generator = LoginDataGenerator()
        training_data = generator.generate_dataset(normal_samples, anomalous_samples)
        
        # Train the model
        detector.train(training_data)
        
        return TrainingResponse(
            message="Model training completed successfully",
            status="success",
            samples_trained=len(training_data)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/detect", response_model=LoginResponse, tags=["Detection"])
async def detect_anomaly(login_request: LoginRequest):
    """Detect if a login attempt is anomalous"""
    if not detector.is_trained:
        raise HTTPException(
            status_code=400, 
            detail="Model not trained. Please train the model first using /model/train"
        )
    
    try:
        # Validate input
        if login_request.login_hour < 0 or login_request.login_hour > 23:
            raise HTTPException(status_code=400, detail="login_hour must be between 0 and 23")
        
        if login_request.device not in ['mobile', 'laptop', 'tablet']:
            raise HTTPException(status_code=400, detail="device must be one of: mobile, laptop, tablet")
        
        # Perform anomaly detection
        result = detector.predict_anomaly(
            login_request.user_id,
            login_request.login_hour,
            login_request.device,
            login_request.location
        )
        
        # Save to MongoDB
        save_login_attempt(result)
        
        return LoginResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/simulate/login", tags=["Simulation"])
async def simulate_login(user_id: Optional[str] = None):
    """Simulate a random login attempt for testing"""
    import random
    
    # Define possible values
    users = ['user_001', 'user_002', 'user_003', 'user_004', 'user_005']
    devices = ['mobile', 'laptop', 'tablet']
    locations = ['USA', 'Canada', 'UK', 'Germany', 'India', 'Singapore', 
                 'Japan', 'South Korea', 'Australia', 'New Zealand']
    
    # Generate random login
    login_request = LoginRequest(
        user_id=user_id or random.choice(users),
        login_hour=random.randint(0, 23),
        device=random.choice(devices),
        location=random.choice(locations)
    )
    
    # Detect anomaly (this will automatically save to MongoDB)
    result = await detect_anomaly(login_request)
    return result

def get_login_history_from_sqlite(limit=50):
    """Get login history from SQLite"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get recent logins, sorted by timestamp (newest first)
        cursor.execute('''
            SELECT user_id, login_hour, device, location, anomaly_score, is_anomaly, status, explanation, timestamp
            FROM login_attempts
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        # Convert to list of dictionaries
        columns = [column[0] for column in cursor.description]
        history = []
        for row in cursor.fetchall():
            history.append(dict(zip(columns, row)))
        
        conn.close()
        return history
    except Exception as e:
        print(f"Error getting history from SQLite: {e}")
        return []

def get_login_stats_from_sqlite():
    """Get login statistics from SQLite"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_logins,
                SUM(CASE WHEN is_anomaly = 1 THEN 1 ELSE 0 END) as anomalous_logins,
                SUM(CASE WHEN is_anomaly = 0 THEN 1 ELSE 0 END) as normal_logins
            FROM login_attempts
        ''')
        
        stats = cursor.fetchone()
        
        # Calculate anomaly rate
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
        print(f"Error getting stats from SQLite: {e}")
        return {"total_logins": 0, "normal_logins": 0, "anomalous_logins": 0, "anomaly_rate_percentage": 0}

@app.get("/history", tags=["History"])
async def get_login_history_endpoint(limit: int = 50):
    """Get recent login history"""
    history = get_login_history_from_sqlite(limit)
    
    if not history:
        return {"message": "No login history available", "history": []}
    
    stats = get_login_stats_from_sqlite()
    
    return {
        "total_logins": stats["total_logins"],
        "returned_logins": len(history),
        "history": history
    }

@app.get("/history/stats", tags=["History"])
async def get_login_stats_endpoint():
    """Get login statistics"""
    stats = get_login_stats_from_sqlite()
    
    if stats["total_logins"] == 0:
        return {"message": "No login history available"}
    
    return stats

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
