# aiml

# 🦸 Multimodal AR Transformation System  

This project is a web-based **augmented reality (AR) filter system** that activates a transformation effect when a user performs a specific **gesture** and **voice command**. Inspired by characters like *Ultraman* and *Kamen Rider*, the system uses **computer vision and speech recognition** to detect the correct inputs and apply the digital transformation.  

## 🚀 Features  
- **Hand Gesture Recognition**: Uses MediaPipe and a custom-trained MLP model to detect the "Let’s Go" pose.  
- **Voice Recognition**: Uses the Vosk API to detect the keyword "Go".  
- **Multimodal Control**: The filter activates only when both the correct gesture and voice command are detected.  
- **Real-time Processing**: Works with a webcam and microphone for seamless interaction.  

## 🛠️ Technologies Used  
- **Python**  
- **MediaPipe** (for hand pose estimation)  
- **Multi-Layer Perceptron (MLP)** (for gesture classification)  
- **Vosk** (for offline speech recognition)  
- **PyAudio** (for microphone input handling)  
- **OpenCV** (for image processing)  

## 📷 How It Works
Start the app – Your webcam and microphone will activate.
Perform the "Let’s Go" gesture – The model will detect your hand pose.
Say "Go" – The speech recognition model will process your voice.
Transformation Activated! – If both inputs match, the AR filter is applied.

## 🎯 Model Training
The gesture recognition model was trained using MediaPipe keypoints stored in keypoint.csv.
It uses a feedforward neural network with the following structure:
Input Layer: 42 nodes
Hidden Layers: 20 and 10 nodes
Output Layer: 3 nodes (gesture classification)
The model was trained for 248 epochs and achieved 96% accuracy.

##🔮 Future Improvements
Add more custom gestures and voice commands.
Improve model accuracy with a larger dataset.
Implement an intuitive UI for filter customization.

##📜 License
This project is licensed under the MIT License.

##🤝 Contributing
Pull requests are welcome! Feel free to open an issue for feature requests or bug fixes.

