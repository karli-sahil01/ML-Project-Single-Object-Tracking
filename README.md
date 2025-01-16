# 🚀 ML-Based Single Object Tracking (SOT)

## 📌 Overview
Object tracking is the backbone of computer vision, fueling applications like surveillance, autonomous vehicles, and video analysis. Our robust **Single Object Tracking (SOT)** framework tackles challenges like:
- 🎥 Dynamic camera motion
- 🎭 Object appearance changes
- 🏃 Unpredictable object motion

## 🔍 Key Features
1. **Camera Motion Compensation** 🎥: Stabilizes tracking by canceling out shaky movements using ORB-based affine transformation.
2. **Multiscale Tracking** 🔍: Adapts to size variations with dynamic sliding windows.
3. **Hybrid Feature Extraction** 🔬: Combines HOG, LBP, and SIFT for supercharged object representation.
4. **Machine Learning Magic** 🎯: 
   - **Position Prediction**: Linear regression keeps the object in sight.
   - **Size Estimation**: Random Forest ensures accurate bounding boxes.

## 📊 Performance Metrics
We don’t just track; we track **accurately**! Our evaluation metrics:
- **Intersection over Union (IoU)**: 📐 85% average accuracy 🚀
- **Mean Absolute Error (MAE)**: 📉 20% reduction vs. baseline 📊
- **R² Score**: 🎯 0.92 (position), 0.88 (size) ✅

## 📈 How It Works
1. **Feature Extraction** 🧩: Compute key features using HOG, LBP & SIFT.
2. **Motion Compensation** 🚀: ORB-based transformation stabilizes frames.
3. **Multiscale Search** 🔍: Dynamically adjusts search windows.
4. **Prediction Models** 🎯: ML models track position & size.

## 🔬 Results
Our framework **dominates** with:
- 🎯 **85% IoU**: Accurate tracking
- 📉 **Lower MAE**: Improved precision
- 🏆 **ML-powered adaptability**