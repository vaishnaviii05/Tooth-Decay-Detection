# 🦷 Tooth Decay Detection Using Deep Learning

Tooth Decay Detection is an AI-based system that uses deep learning and image processing to detect dental caries from dental images. The model analyzes uploaded images using a CNN and predicts whether decay is present or not, helping in early diagnosis, faster screening, and improved dental healthcare support.

## 🚀 Features

Tooth decay detection from dental images

CNN-based deep learning model

Flask web app for real-time prediction

Image preprocessing and evaluation tools

Visualization and explainability modules (XAI)

Clean and reproducible project structure

## 🧠 Technology Stack

Python

TensorFlow / Keras

OpenCV

NumPy, Matplotlib

Flask

Scikit-learn

## 📁 Project Structure
tooth-decay-detection/
 ├── app.py
 ├── train_from_archive.py
 ├── train_from_archive_updated.py
 ├── model_utils.py
 ├── confusion_matrix_eval.py
 ├── visualize_model_working.py
 ├── visualize_model_xai.py
 ├── requirements.txt
 ├── README.md
 ├── templates/
 │    └── index.html
 └── static/
      └── style.css

## 📦 Model & Dataset

Due to GitHub file size limits, the trained model and dataset are not included in this repository.

🔗 Download Pretrained Model:

- https://drive.google.com/file/d/1RDXWTtr_FMDtccFw2RBAl7hwkocPqDvL/view?usp=drive_link
- https://drive.google.com/file/d/105tdClZVw7V6e7HBoDB4U0lDZIY_I8SS/view?usp=drive_link

After downloading, place the model file in the project root directory.

▶️ How to Run This Project Locally
1️⃣ Clone the Repository
git clone https://github.com/vaishnaviii05/tooth-decay-detection.git
cd tooth-decay-detection

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Download Pretrained Model

Download the model from the link above and place it in the project folder.

5️⃣ Run the Flask App
python app.py

6️⃣ Open in Browser

Open:

http://127.0.0.1:5000/

🧪 To Train the Model Again (Optional)
python train_from_archive.py

## 📊 Evaluation

- Confusion matrix evaluation supported

- Visualization of predictions

- Explainable AI (XAI) visualizations available

## 📸 Demo
- Uplaoded Image
  <img width="2943" height="1435" alt="372" src="https://github.com/user-attachments/assets/d8807832-7377-4b9e-a3f6-5f52f6b4a35d" />
<img width="2943" height="1435" alt="360" src="https://github.com/user-attachments/assets/681a88f0-6309-4f9a-b0e7-411256e96b08" />
<img width="2943" height="1435" alt="354" src="https://github.com/user-attachments/assets/31e35f68-8606-4857-bd44-3603143bc034" />
<img width="2943" height="1435" alt="347" src="https://github.com/user-attachments/assets/dce9e35d-88a5-47da-8296-5b8d455c5868" />
<img width="2943" height="1435" alt="318" src="https://github.com/user-attachments/assets/fcd07158-f647-45bf-a584-6c4e222d6711" />
<img width="2943" height="1435" alt="306" src="https://github.com/user-attachments/assets/c036d0fb-3acf-4b84-850b-8ae56a65389f" />
<img width="2943" height="1435" alt="761" src="https://github.com/user-attachments/assets/e230c165-d7de-403a-97bc-5526c523de8a" />


- Results Image
<img width="1150" height="193" alt="image" src="https://github.com/user-attachments/assets/85ad9ba9-39b1-4b34-b26f-235ac82c047b" />
<img width="2943" height="1435" alt="original" src="https://github.com/user-attachments/assets/fed5fe68-d404-4358-9538-7f1d01f765c2" />
<img width="2943" height="1435" alt="mask" src="https://github.com/user-attachments/assets/d54135f4-2b27-48fe-ae6c-25f534c07dc4" />
<img width="2943" height="1435" alt="gradcam" src="https://github.com/user-attachments/assets/816fa373-84b1-45b0-a0f0-edf925615448" />
<img width="2943" height="1435" alt="boxes" src="https://github.com/user-attachments/assets/ff2423fd-700f-466c-ad66-ebef59052957" />



## 👩‍💻 Author

Vaishnavi Singh
MCA | Machine Learning & Web Development
GitHub: https://github.com/vaishnaviii05

## 📜 License

This project is developed for educational and academic purposes.
