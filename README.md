# 🔐 AI-Based Login Anomaly Detection System

A machine learning system that detects suspicious login attempts by analyzing user behavior patterns such as time, device, and location.

---

## 📌 Overview

This system identifies anomalous login activity using unsupervised learning techniques. It helps detect potential account compromise without relying on predefined attack signatures.

---

## 🧠 Approach

* Behavioral feature extraction:

  * Login time patterns
  * Device type
  * Location/IP encoding
* Model learns normal user behavior
* Deviations are flagged as anomalies

---

## ⚙️ Tech Stack

* **Language:** Python
* **ML:** scikit-learn (Isolation Forest)
* **Backend:** FastAPI
* **Frontend:** Streamlit
* **Database:** SQLite

---

## 📊 Key Features

* 🚨 Real-time anomaly detection
* 📉 Anomaly score-based classification
* 📊 Visual login pattern analysis (Streamlit UI)
* 🧠 Unsupervised learning (no labeled data required)

---

## 📈 Performance & Metrics

* Model: Isolation Forest
* Dataset size: ~1000+ login records
* Features used: time, device, location
* Detection threshold tuned to reduce false positives
* Achieved stable anomaly detection across varied patterns

---

## 🔗 Live Demo

👉 [(Deployed link here)](https://anomalydetection-geu44sjyunzueygohtzyu5.streamlit.app/)

---

## 🧪 Example Use Case

* Detect unusual login time
* Flag new device usage
* Identify suspicious location changes

---

## 🚧 Future Improvements

* GeoIP-based location tracking
* Deep learning-based anomaly detection
* Integration with authentication systems

---

## 👨‍💻 Author

Aryan Shukla
