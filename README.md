# FACE RECOGNITION
### Face Recognition System

## Technologies Used

- Python 3.x
- OpenCV
- Dlib (Optional: for improved accuracy and additional features)
- numpy
- face_recognition library (built on top of dlib)

## Features

- **Face Detection**: Identifies and marks faces in images or videos.
- **Face Recognition**: Compares detected faces against a stored database of known faces to identify individuals.
- **Real-time Video Recognition**: Detect and recognize faces in a live video feed.
- **Support for Multiple Faces**: Can handle multiple faces in a single image or video frame.

## Installation

### Prerequisites

Make sure you have Python 3.x installed on your machine.

### Steps to Install

1. Clone the repository:

    ```bash
    git clone https://github.com/Vighnesh3006/face_recognition_project.git
    cd face-recognition
    ```

2. Install dependencies:

    You can install the required libraries by running the following command:

    ```bash
    pip install -r requirements.txt
    ```    
3. Place the images of known faces in the `known_faces/` directory. These images will be used to train the recognition model.

4. (Optional) To use your own webcam for real-time face recognition, make sure to have a webcam connected to your computer.

## Usage

### Training the Model

To train the model with your known faces, use the following command:


python train_faces.py

## contact

Vighneshghorpade007@gmail.com
Vighnesh3006
