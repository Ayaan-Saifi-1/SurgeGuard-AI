# SurgeGuard AI 🛡️

**Intelligent Crowd Surge Prediction & Management System**
*Developed for Synapse AI Hackathon 2026 @ DTU*

SurgeGuard AI uses computer vision (YOLOv8) to monitor crowd density in real-time. It transforms standard CCTV feeds into actionable safety data, triggering alerts *before* a stampede occurs.

## 🚨 The Problem
High-density events (stadiums, concerts) in India are prone to critical crowd surges. Manual monitoring is reactive and often too slow to prevent tragedies like the 2025 RCB Victory Parade incident.

## 💡 The Solution
A proactive system that:
- **Detects** humans in real-time with high accuracy.
- **Analyzes** crowd density per frame.
- **Alerts** security instantly when thresholds are breached.

## 🚀 Features
- **Real-time Detection:** Powered by YOLOv8 Nano for low-latency inference.
- **Dynamic Thresholding:** Adjustable safety limits.
- **Visual Dashboard:** Live heatmap overlay with "Safe/Critical" status indicators.
- **Privacy First:** Processes video locally; no facial recognition data is stored.

## 🛠️ Tech Stack
- **Language:** Python 3.10
- **Vision Model:** YOLOv8 (Ultralytics)
- **Image Processing:** OpenCV & CVZone
- **Hardware:** Runs on standard CPU (No heavy GPU required).

## ⚙️ Installation & Usage

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/YOUR-USERNAME/SurgeGuard-AI.git](https://github.com/YOUR-USERNAME/SurgeGuard-AI.git)
   cd SurgeGuard-AI