# AI-based Login Anomaly Detection System

A comprehensive cybersecurity project that uses machine learning to detect suspicious login attempts based on user behavior patterns.

## 🎯 Project Overview

This system monitors user login behavior (time, device, location) and detects anomalous login attempts using an Isolation Forest algorithm. It provides real-time anomaly detection with detailed explanations for security teams.

## 🏗️ Project Structure

```
login_anomaly_detection/
├── data/
│   └── data_generator.py      # Synthetic login data generation
├── models/
│   └── anomaly_detector.py   # Isolation Forest model
├── utils/
│   └── data_processor.py      # Data preprocessing and encoding
├── api/
│   └── main.py               # FastAPI backend
├── requirements.txt           # Python dependencies
└── README.md                # Project documentation
```

## 🚀 Features

- **Real-time Anomaly Detection**: Detects suspicious login attempts instantly
- **Explainable AI**: Provides detailed reasons for anomaly detection
- **Multiple Anomaly Types**: Identifies unusual time, device, location, or combined patterns
- **RESTful API**: Clean FastAPI endpoints for integration
- **Synthetic Data Generation**: Realistic training data with normal and anomalous patterns
- **Modular Architecture**: Clean separation of concerns

## 📊 Data Schema

### Login Data Fields
- `user_id`: Unique user identifier
- `login_hour`: Hour of day (0-23)
- `device`: Device type (mobile, laptop, tablet)
- `location`: Geographic location (country/region)
- `is_anomaly`: Ground truth label (for training)

## 🤖 Machine Learning Model

**Algorithm**: Isolation Forest
- Unsupervised anomaly detection
- Handles high-dimensional data
- Provides anomaly scores
- Contamination parameter for expected anomaly rate

## 🔧 Installation & Setup

### 1. Clone and Navigate
```bash
cd login_anomaly_detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python -c "
from data.data_generator import LoginDataGenerator
from models.anomaly_detector import AnomalyDetector

# Generate training data
generator = LoginDataGenerator()
training_data = generator.generate_dataset(normal_samples=1000, anomalous_samples=200)

# Train model
detector = AnomalyDetector()
detector.train(training_data)
print('Model training completed!')
"
```

### 4. Start the API Server
```bash
cd api
python main.py
```

The API will be available at `http://localhost:8000`

## 📡 API Endpoints

### Model Management
- `POST /model/train` - Train the anomaly detection model
- `GET /model/status` - Check model training status

### Anomaly Detection
- `POST /detect` - Detect anomalies in login attempts
- `POST /simulate/login` - Simulate random login for testing

### History & Analytics
- `GET /history` - Get recent login history
- `GET /history/stats` - Get login statistics

### Health & Info
- `GET /` - API information and endpoints
- `GET /health` - Health check

## 🧪 Usage Examples

### 1. Train the Model
```bash
curl -X POST "http://localhost:8000/model/train?normal_samples=1000&anomalous_samples=200"
```

### 2. Detect Anomaly
```bash
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "login_hour": 3,
    "device": "mobile",
    "location": "Russia"
  }'
```

### 3. Simulate Login
```bash
curl -X POST "http://localhost:8000/simulate/login?user_id=user_001"
```

## 📈 Sample Responses

### Normal Login
```json
{
  "user_id": "user_001",
  "login_hour": 10,
  "device": "laptop",
  "location": "USA",
  "anomaly_score": 0.123,
  "is_anomaly": false,
  "status": "normal",
  "explanation": null
}
```

### Suspicious Login
```json
{
  "user_id": "user_001",
  "login_hour": 3,
  "device": "tablet",
  "location": "Russia",
  "anomaly_score": -0.456,
  "is_anomaly": true,
  "status": "suspicious",
  "explanation": "Unusual login time (3:00), Rare device usage (tablet), Unusual location (Russia)"
}
```

## 🔍 Anomaly Detection Logic

The system identifies anomalies based on:

1. **Unusual Time**: Login hours significantly different from user's normal pattern
2. **New Device**: Devices rarely used by the user
3. **Unusual Location**: Geographic locations not commonly accessed
4. **Combined Patterns**: Multiple unusual factors together

## 🛡️ Security Features

- **Real-time Processing**: Instant anomaly detection
- **Explainable Results**: Clear reasons for security teams
- **Configurable Sensitivity**: Adjustable contamination parameter
- **Audit Trail**: Complete login history tracking
- **Modular Design**: Easy integration with existing systems

## 📊 Model Performance

The Isolation Forest model provides:
- **Precision**: High accuracy in anomaly detection
- **Recall**: Effective identification of suspicious patterns
- **F1-Score**: Balanced performance metrics
- **Anomaly Scores**: Continuous risk assessment

## 🔄 Workflow

1. **Data Generation**: Create synthetic training data with realistic patterns
2. **Preprocessing**: Encode categorical features and scale numerical data
3. **Model Training**: Train Isolation Forest on normal and anomalous patterns
4. **Real-time Detection**: Process login attempts and calculate anomaly scores
5. **Explanation Generation**: Provide human-readable reasons for anomalies
6. **Response**: Return detailed results with risk assessment

## 🚀 Production Considerations

For production deployment:

1. **Database Integration**: Replace in-memory storage with persistent database
2. **Authentication**: Add API key or JWT authentication
3. **Rate Limiting**: Implement rate limiting for API endpoints
4. **Monitoring**: Add logging and monitoring
5. **Scaling**: Deploy with containerization (Docker/Kubernetes)
6. **Data Pipeline**: Real-time data ingestion from authentication systems

## 🧪 Testing

### Test Normal Login
```bash
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "login_hour": 10,
    "device": "laptop",
    "location": "USA"
  }'
```

### Test Anomalous Login
```bash
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "login_hour": 2,
    "device": "tablet",
    "location": "Brazil"
  }'
```

## 📚 Technical Details

### Dependencies
- **FastAPI**: Modern web framework for APIs
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **pydantic**: Data validation

### Model Parameters
- **Algorithm**: Isolation Forest
- **Contamination**: 0.1 (10% expected anomalies)
- **Random State**: 42 (reproducible results)
- **Features**: login_hour, device_encoded, location_encoded

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational and demonstration purposes.

## 🔮 Future Enhancements

- **Real User Data Integration**: Connect to actual authentication logs
- **Advanced Models**: Experiment with deep learning approaches
- **Time Series Analysis**: Incorporate temporal patterns
- **Geographic IP Validation**: Verify location against IP addresses
- **User Behavior Profiling**: Individual user baseline patterns
- **Alert System**: Integration with security monitoring tools
- **Dashboard**: Web interface for visualization and analysis
