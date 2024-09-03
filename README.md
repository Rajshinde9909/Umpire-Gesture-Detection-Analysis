# Umpire Hand Gesture Detection and Analysis

## Project Overview

This project focuses on detecting and analyzing umpire hand gestures during sports events using camera footage. By employing computer vision techniques, we aim to recognize specific hand gestures made by umpires, translate these gestures into meaningful signals or decisions, and visualize the data using line graphs to identify trends, frequency, and patterns.

## Features

- **Real-time Gesture Detection**: Capture and process video footage to detect umpire hand gestures.
- **Gesture Recognition**: Identify specific gestures and translate them into corresponding signals or decisions.
- **Data Logging**: Record recognized gestures for further analysis.
- **Visualization**: Generate line graphs to visualize trends, frequency, and patterns of umpire gestures over time.
- **Performance Analysis**: Analyze the accuracy and consistency of umpire decisions.

## Prerequisites

- Python 3
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- TensorFlow/Keras (for deep learning models)
- Jupyter Notebook (optional, for development and testing)
  
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/umpire-gesture-detection.git
   cd umpire-gesture-detection
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
## How It Works

### Gesture Recognition

1. **Video Capture:** The application captures live video from the webcam using OpenCV.

2. **Hand Detection:** The captured frames are processed by Mediapipe's Hands solution to detect hand landmarks.

3. **Gesture Classification:** The detected hand landmarks are analyzed to classify gestures based on the position of the wrist landmark in different regions of the frame. The application is pre-configured to recognize five specific cricket umpire gestures:
   - **Out**
   - **Four**
   - **Six**
   - **New Ball**
   - **Wide Ball**

4. **Displaying Results:** Recognized gestures are displayed on the Tkinter interface in real-time, and the corresponding counts are updated in the gesture history.

5. **Graphical Visualization:** A graph of the gesture history is displayed using Matplotlib when the "Show Graph" button is clicked.

### UI Controls

- **Recognize Gesture Button:** Starts the gesture recognition process. Once clicked, the application continuously monitors and detects gestures from the live video feed.

- **Show Graph Button:** Displays a graph of the recognized gesture history, showing the counts of each gesture over time.
