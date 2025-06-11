# 🛡️ AI-Powered Proctoring System

A lightweight, offline-capable AI-based proctoring system that monitors student behavior during online assessments. Built using OpenCV and facial landmark detection techniques, the system detects signs of cheating such as eye movement, face turning, and mouth activity.

## 🚀 Features

- 🎥 Real-time webcam monitoring
- 👁️ Eye gaze and face direction tracking
- 👄 Mouth activity detection (e.g., talking)
- 📊 Logs violations with timestamps (CSV or MongoDB)
- 💾 Offline-capable and resource-efficient
- 🔐 Privacy-respecting, runs locally on the device

## 🧠 Tech Stack

- Python
- OpenCV
- MediaPipe(facial landmark detection)
- CSV (for violation logging)
- Flask (optional UI or integration)

- ## 📈 How It Works

1. The system captures webcam input using OpenCV.
2. Facial landmarks are detected to track eye movement, head pose, and mouth.
3. Any suspicious activity (e.g., looking away for too long, talking) is flagged.
4. Violations are recorded in a CSV or MongoDB with timestamps and event type.

## 🛠️ Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/proctoring-system.git
   cd proctoring-system
   ```
