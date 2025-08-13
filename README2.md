🫁 Lung Disease Detection from Chest X-Rays (CNN + VGG16 Transfer Learning)
📌 Project Overview
This project focuses on automating the diagnosis of lung diseases such as Pneumonia, Tuberculosis, and other conditions using chest X-ray images.
We implemented a Convolutional Neural Network (CNN) with VGG16 transfer learning to achieve high accuracy in classification.
The model is designed to assist radiologists by providing faster and more accurate diagnoses from medical images.

🚀 Features
Image-based diagnosis of multiple lung disease classes.

Transfer Learning using pre-trained VGG16.

Data augmentation for better generalization.

Achieves around 80% accuracy on test data.

Easily extendable for new lung disease categories.

Modular code with separate scripts for training and prediction.

📂 Dataset
The dataset contains chest X-ray images categorized into 5 classes:

Pneumonia

Tuberculosis

COVID-19

Normal

Other lung conditions

📌 Dataset Source: Kaggle - Lungs Disease Dataset (4 types)

🛠️ Tech Stack
Python 3.8+

TensorFlow / Keras

NumPy, Pandas

Matplotlib, Seaborn

scikit-learn

📜 Installation
bash
Copy
Edit
# Clone the repository
git clone https://github.com/your-username/lung-disease-detection.git
cd lung-disease-detection

# Install dependencies
pip install -r requirements.txt
📁 Project Structure
graphql
Copy
Edit
lung-disease-detection/
│
├── train.py         # Train the CNN model
├── predict.py       # Predict disease from new chest X-ray images
├── utils.py         # Helper functions (data loading, preprocessing, etc.)
├── requirements.txt # Required dependencies
├── README.md        # Project documentation
└── saved_model/     # Trained model files
🏋️‍♂️ Training
bash
Copy
Edit
python train.py
Trains the model on the dataset.

Saves the trained model to saved_model/.

🔍 Prediction
bash
Copy
Edit
python predict.py --image_path sample_xray.jpg
Loads the trained model.

Predicts the lung disease type from the given image.

📊 Results
Accuracy: ~80% on test set

Model: VGG16 (Transfer Learning)

Loss Curve & Accuracy Curve:


📌 Future Improvements
Improve accuracy with EfficientNet or ResNet.

Implement Grad-CAM for visual explanations.

Deploy as a web app using Streamlit or Flask.

👩‍💻 Author
Evanjelin Nisha
📧 evanjelin4215@gmail.com
🌐 [LinkedIn](https://www.linkedin.com/in/evanjelin-nisha-602283280/) 

