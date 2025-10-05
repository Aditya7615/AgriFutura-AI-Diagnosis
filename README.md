AgriFutura AI Diagnosis 🌿
An AI-powered web application for the accurate and real-time detection of plant diseases. This project leverages a deep learning model to classify plant leaf images, helping farmers identify issues early and improve crop health.

Note: You should replace the image link above with a real screenshot of your running application!

📋 Table of Contents
Project Overview
Key Features
Technology Stack
Model Architecture
Setup and Installation
How to Use
Dataset
Future Improvements
License
Contact
📖 Project Overview
Plant diseases pose a significant threat to global food security, causing substantial losses in crop yield. Traditional methods of disease detection are often manual, time-consuming, and require expert knowledge.

AgriFutura AI Diagnosis provides a modern solution by using a powerful Convolutional Neural Network (CNN) to automatically identify diseases from images of plant leaves. The user-friendly web interface, built with Streamlit, allows anyone to upload an image and receive an instant diagnosis along with actionable advice.

✨ Key Features
Accurate Disease Identification: Utilizes a robust deep learning model trained on thousands of images for high-precision classification.
User-Friendly Web Interface: A simple and intuitive UI built with Streamlit for easy image uploads and clear results.
Real-Time Diagnosis: Get instant predictions and confidence scores within seconds.
Actionable Advice: Provides helpful tips and treatment suggestions for the identified diseases.
Scalable Solution: The use of the ONNX format for the model ensures efficient and cross-platform deployment capabilities.
🛠️ Technology Stack
Backend: Python
Deep Learning: TensorFlow, Keras
Model Deployment: ONNX Runtime
Web Framework: Streamlit
Data Handling: NumPy, Pillow (PIL)
🧠 Model Architecture
The core of this application is a Convolutional Neural Network (CNN) trained for image classification. The model was developed using TensorFlow/Keras and then converted to the ONNX (Open Neural Network Exchange) format. This conversion allows for high-performance inference on various platforms, making the application fast and portable.

⚙️ Setup and Installation
Follow these steps to set up and run the project on your local machine.

Prerequisites
Git
Python 3.10 or newer
Git LFS (for handling the large model file)
Step-by-Step Guide
1. Clone the Repository:

Bash
git clone https://github.com/Aditya7615/AgriFutura-AI-Diagnosis.git
cd AgriFutura-AI-Diagnosis
2. Set up Git LFS:
This project uses Git Large File Storage (LFS) to manage the ONNX model file. You must have Git LFS installed to pull the model correctly.

Bash
# Install Git LFS (if you haven't already)
git lfs install

# Pull the large files from LFS storage
git lfs pull
3. Create a Virtual Environment (Recommended):
It's best practice to create a virtual environment to manage project dependencies.

Bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
4. Install Dependencies:
Install all the required Python libraries from the requirements.txt file.

Bash
pip install -r requirements.txt
5. Run the Streamlit Application:
You're all set! Launch the application using the following command:

Bash
streamlit run app.py
Your web browser should automatically open to the application's URL (usually http://localhost:8501).

🚀 How to Use
Launch the App: Run the streamlit run app.py command.
Upload an Image: Click the "Browse files" button and select a .jpg, .jpeg, or .png image of a plant leaf.
Get a Diagnosis: The application will process the image and display the predicted disease, a confidence score, and recommended actions.
📊 Dataset
The model was trained on the PlantVillage Dataset, which contains thousands of images of healthy and diseased plant leaves across numerous species. This diverse dataset enables the model to generalize well to new images.

You can find the dataset on Kaggle: PlantVillage Dataset

🔮 Future Improvements
[ ] Deploy the application to a cloud service like Streamlit Community Cloud or Hugging Face Spaces for public access.
[ ] Develop a mobile application version using TensorFlow Lite for on-field diagnosis.
[ ] Expand the dataset to include more plant species and diseases.
[ ] Create a REST API endpoint to serve model predictions to other applications.
📜 License
This project is licensed under the MIT License. See the LICENSE file for more details.

👨‍💻 Contact
Aditya Goyal - GitHub - LinkedIn ---
