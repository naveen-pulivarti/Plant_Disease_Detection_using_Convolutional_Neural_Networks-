# Plant Disease Detection from Images

## Objective
Develop a Streamlit application that enables users to upload images of plant leaves and accurately predict the presence and type of plant disease using a Convolutional Neural Network (CNN) model.

## Project Scope
- **End-to-End Development:** Responsible for all project components, including data preprocessing, CNN training, and building a functional application.
- **Real-World Relevance:** Provide a tool for farmers and gardeners to quickly diagnose plant diseases.

## Key Components

### 1. Image Preprocessing
- **Data Preparation:** Implement preprocessing steps like resizing and normalization.
- **Dataset:** Utilize the New Plant Diseases Dataset from Kaggle.

### 2. Disease Classification
- **CNN Model:** Define and train a custom CNN architecture for 30 epochs, achieving good evaluation scores.
- **Transfer Learning:** Evaluate pretrained models (ResNet50, EfficientNetB0, DenseNet121) without modifications, then apply transfer learning by freezing layers and retraining the last layer for improved performance.

### 3. Performance and Optimization
- **Model Evaluation:** Assess performance using accuracy, precision, and recall metrics.
- **Optimization:** Ensure minimal latency for real-time predictions.

### 4. User Interface Development
- **Streamlit Application:** Develop a web interface that allows users to upload images of plant leaves and receive disease predictions.
- **Usability:** Ensure an intuitive experience with clear instructions and feedback.

### 5. Testing
- **Testing:** Verify prediction accuracy and robustness with various image inputs.

## Expected Results
- **Functional Application:** A user-friendly Streamlit web application for plant disease detection.
- **Model Performance Report:** Detailed evaluation of the CNN model’s performance metrics.
- **User Guide:** Documentation for application usage and feedback.

## Tools and Technologies
- **Programming Language:** Python
- **Frameworks and Libraries:** Streamlit, OpenCV, TensorFlow/Keras or PyTorch
- **Dataset:** New Plant Diseases Dataset from Kaggle

## Deliverables
1. Streamlit Application
2. Well-documented Python codebase
3. Trained CNN models
4. Comprehensive project report
5. User guide for application setup and usage

## Project Guidelines
- **Independent Work:** Complete all components independently to demonstrate proficiency in CNNs, image processing, and web development.
- **Final Presentation:** Present the application and its functionality to faculty.

This project offers a valuable opportunity to apply CNN techniques to automate plant disease detection, providing practical benefits in agriculture.
