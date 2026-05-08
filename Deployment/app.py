import streamlit as st 
import pickle 
import os
import torch
import torch.nn as nn
import warnings
import numpy as np 
from PIL import Image
import torchvision.transforms as transforms
from joblib import load
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler, RobustScaler

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

st.set_page_config(page_title="Mulitple Disease Prediction",layout="wide", page_icon="👨‍🦰🤶")

working_dir = os.path.dirname(os.path.abspath(__file__))

# Load models with proper path handling
try:
    diabetes_model = load(os.path.join(working_dir, 'saved_models', 'diabetes.joblib'))
except Exception as e:
    st.error(f"Error loading diabetes model: {str(e)}")
    diabetes_model = None

try:
    heart_disease_model = load(os.path.join(working_dir, 'saved_models', 'heart.joblib'))
except Exception as e:
    st.error(f"Error loading heart disease model: {str(e)}")
    heart_disease_model = None

try:
    kidney_disease_model = load(os.path.join(working_dir, 'saved_models', 'kidney.joblib'))
except Exception as e:
    st.error(f"Error loading kidney disease model: {str(e)}")
    kidney_disease_model = None

# Load model brain tumor image classification to MRI & CT Scan


import torch
import torch.nn as nn

class MRIModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),
            nn.MaxPool2d(4, 4),

            nn.Flatten(),

            nn.Linear(7 * 7 * 1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.layer_stack(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MRIModel(in_channels=3, num_classes=2)

model.load_state_dict(
    torch.load("C:\\Users\\lap shop\\OneDrive\\المستندات\\ML_Projects\\Multiple_Disease_Prediction-main\\Brain Tumor images\\saved_model\\model.pth", map_location=device)
)

model.to(device)
model.eval()


# Load the scalers for diabetes (if they exist)
try:
    diabetes_robust_scaler = load(os.path.join(working_dir, 'saved_models', 'diabetes_robust_scaler.joblib'))
    diabetes_standard_scaler = load(os.path.join(working_dir, 'saved_models', 'diabetes_standard_scaler.joblib'))
except Exception as e:
    diabetes_robust_scaler = None
    diabetes_standard_scaler = None
    st.warning(f"Could not load diabetes scalers: {str(e)}")






NewBMI_Overweight=0
NewBMI_Underweight=0
NewBMI_Obesity_1=0
NewBMI_Obesity_2=0 
NewBMI_Obesity_3=0
NewInsulinScore_Normal=0 
NewGlucose_Low=0
NewGlucose_Normal=0 
NewGlucose_Overweight=0
NewGlucose_Secret=0




with st.sidebar:
    selected = option_menu("Mulitple Disease Prediction", 
                ['Diabetes Prediction',
                 'Heart Disease Prediction',
                 'Kidney Disease Prediction',
                 'Brain Tumor Prediction'],
                 menu_icon='hospital-fill',
                 icons=['activity','heart', 'person' , 'cpu'],
                 default_index=0)




### Diabetes Disease

if selected == 'Diabetes Prediction':
    st.title("Diabetes Detection Using Machine Learning")

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
    with col2:
        Glucose = st.text_input("Glucose Level")
    with col3:
        BloodPressure = st.text_input("BloodPressure Value")
    with col1:
        SkinThickness = st.text_input("SkinThickness Value")
    with col2:
        Insulin = st.text_input("Insulin Value")
    with col3:
        BMI = st.text_input("BMI Value")
    with col1:
        DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction Value")
    with col2:
        Age = st.text_input("Age")


    # code for Prediction

    diabetes_result = ""


    # creating a button for Prediction 
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
      if st.button("Diabetes Test Result"):
        # Validate that all fields are filled
        required_fields = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        if any(field == '' or field is None for field in required_fields):
            st.error('⚠️ Please fill in all fields before prediction')
        else:
            # First, create the 8 original features for RobustScaler
            original_features = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), 
                                float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]
            
            # Apply RobustScaler to the original 8 features
            if diabetes_robust_scaler is not None:
                original_features_scaled = diabetes_robust_scaler.transform([original_features])[0]
            else:
                original_features_scaled = original_features
            
            # Now apply feature engineering on the original (unscaled) values
            if float(BMI)<=18.5:
                NewBMI_Underweight = 1
            elif 18.5 < float(BMI) <=24.9:
                pass
            elif 24.9<float(BMI)<=29.9:
                NewBMI_Overweight =1
            elif 29.9<float(BMI)<=34.9:
                NewBMI_Obesity_1 =1
            elif 34.9<float(BMI)<=39.9:
                NewBMI_Obesity_2=1
            elif float(BMI)>39.9:
                NewBMI_Obesity_3 = 1
            
            if 16<=float(Insulin)<=166:
                NewInsulinScore_Normal = 1

            if float(Glucose)<=70:
                NewGlucose_Low = 1
            elif 70<float(Glucose)<=99:
                NewGlucose_Normal = 1
            elif 99<float(Glucose)<=126:
                NewGlucose_Overweight = 1
            elif float(Glucose)>126:
                NewGlucose_Secret = 1

            # Combine scaled original features with engineered features
            user_input = list(original_features_scaled) + [NewBMI_Underweight,
                        NewBMI_Overweight, NewBMI_Obesity_1,
                        NewBMI_Obesity_2, NewBMI_Obesity_3, NewInsulinScore_Normal, 
                        NewGlucose_Low, NewGlucose_Normal, NewGlucose_Overweight,
                        NewGlucose_Secret]
            
            # Apply StandardScaler to all 18 features
            if diabetes_standard_scaler is not None:
                user_input_scaled = diabetes_standard_scaler.transform([user_input])
                prediction = diabetes_model.predict(user_input_scaled)
            else:
                # Fallback if scalers not available
                scaler = StandardScaler()
                user_input_scaled = scaler.fit_transform([user_input])
                prediction = diabetes_model.predict(user_input_scaled)
            
            if prediction[0]==1:
                diabetes_result = "❌ The Person has diabetic"
                st.error(diabetes_result)
            else:
                diabetes_result = "✅ The Person has no diabetic"
                st.success(diabetes_result)




### Heart Disease

if selected == 'Heart Disease Prediction':
    st.title("Heart Disease Detection Using Machine Learning")
    col1, col2, col3  = st.columns(3)

    with col1:
        age = st.text_input("Age")
    with col2:
        sex = st.text_input("Sex")
    with col3:
        cp = st.text_input("Chest Pain Types")
    with col1:
        trestbps = st.text_input("Resting Blood Pressure")
    with col2:
        chol = st.text_input("Serum Cholestroal in mg/dl")
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')



    # code for Prediction

    heart_disease_result = ""



    # creating a button for Prediction
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
      if st.button("Heart Disease Test Result"):
        # Validate that all fields are filled
        required_fields = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
        if any(field == '' or field is None for field in required_fields):
            st.error('⚠️ Please fill in all fields before prediction')
        else:
            user_input = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
            user_input = [float(x) for x in user_input]
            
            # Heart Disease model was trained WITHOUT scaling
            prediction = heart_disease_model.predict([user_input])
            if prediction[0]==1:
                heart_disease_result = "❌ This Person is having heart disease"
                st.error(heart_disease_result)
            else:
                heart_disease_result = "✅ This Person does not have any heart disease"
                st.success(heart_disease_result)



### Kidney Disease

if selected == 'Kidney Disease Prediction':
    
    st.title("Kidney Disease Detection using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        age = st.text_input('Age')

    with col2:
        blood_pressure = st.text_input('Blood Pressure')

    with col3:
        specific_gravity = st.text_input('Specific Gravity')

    with col4:
        albumin = st.text_input('Albumin')

    with col5:
        sugar = st.text_input('Sugar')

    with col1:
        red_blood_cells = st.text_input('Red Blood Cell')

    with col2:
        pus_cell = st.text_input('Pus Cell')

    with col3:
        pus_cell_clumps = st.text_input('Pus Cell Clumps')

    with col4:
        bacteria = st.text_input('Bacteria')

    with col5:
        blood_glucose_random = st.text_input('Blood Glucose Random')

    with col1:
        blood_urea = st.text_input('Blood Urea')

    with col2:
        serum_creatinine = st.text_input('Serum Creatinine')

    with col3:
        sodium = st.text_input('Sodium')

    with col4:
        potassium = st.text_input('Potassium')

    with col5:
        haemoglobin = st.text_input('Haemoglobin')

    with col1:
        packed_cell_volume = st.text_input('Packet Cell Volume')

    with col2:
        white_blood_cell_count = st.text_input('White Blood Cell Count')

    with col3:
        red_blood_cell_count = st.text_input('Red Blood Cell Count')

    with col4:
        hypertension = st.text_input('Hypertension')

    with col5:
        diabetes_mellitus = st.text_input('Diabetes Mellitus')

    with col1:
        coronary_artery_disease = st.text_input('Coronary Artery Disease')

    with col2:
        appetite = st.text_input('Appetitte')

    with col3:
        peda_edema = st.text_input('Peda Edema')
    with col4:
        aanemia = st.text_input('Aanemia')



    # code for Prediction

    kindey_diagnosis = ''

    # creating a button for Prediction
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Kidney's Test Result"):
            # Validate that all fields are filled
            required_fields = [age, blood_pressure, specific_gravity, albumin, sugar,
               red_blood_cells, pus_cell, pus_cell_clumps, bacteria,
               blood_glucose_random, blood_urea, serum_creatinine, sodium,
               potassium, haemoglobin, packed_cell_volume,
               white_blood_cell_count, red_blood_cell_count, hypertension,
               diabetes_mellitus, coronary_artery_disease, appetite,
               peda_edema, aanemia]
            
            if any(field == '' or field is None for field in required_fields):
                st.error('⚠️ Please fill in all fields before prediction')
            else:
                user_input = [age, blood_pressure, specific_gravity, albumin, sugar,
                   red_blood_cells, pus_cell, pus_cell_clumps, bacteria,
                   blood_glucose_random, blood_urea, serum_creatinine, sodium,
                   potassium, haemoglobin, packed_cell_volume,
                   white_blood_cell_count, red_blood_cell_count, hypertension,
                   diabetes_mellitus, coronary_artery_disease, appetite,
                   peda_edema, aanemia]

                user_input = [float(x) for x in user_input]

                # Kidney Disease model was trained WITHOUT scaling
                prediction = kidney_disease_model.predict([user_input])

                if prediction[0] == 1:
                    kindey_diagnosis = "❌ The Person has Kidney's disease"
                    st.error(kindey_diagnosis)
                else:
                    kindey_diagnosis = "✅ The Person does not have Kidney's disease"
                    st.success(kindey_diagnosis)




# Brain Tumor Prediction
if selected == 'Brain Tumor Prediction':
    st.title("Brain Tumor Detection using MRI & CT Scan")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        #st.markdown("### 🧠 Brain Tumor Detection")

        uploaded_file = st.file_uploader(
        "Upload MRI/CT Image",
        type=["jpg", "png", "jpeg"],
        key="mri_uploader"
    )
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, width=500)

            # preprocessing
            image_tensor = transform(image)
            image_tensor = image_tensor.unsqueeze(0).to(device)


            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
            with btn_col2:
                predict_clicked = st.button("Predict", key="predict_btn")


        if uploaded_file is not None and predict_clicked:
            with torch.no_grad():
                output = model(image_tensor)
                prediction = torch.argmax(output, dim=1).item()

            if prediction == 1:
                st.error("❌ Brain Tumor Detected")
            else:
                st.success("✅ No Brain Tumor")