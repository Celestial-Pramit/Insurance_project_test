import gradio as gr
import pickle
import numpy as np
import pandas as pd

# ================================
# LOAD TRAINED MODEL
# ================================
model = pickle.load(open("insurance_model.pkl", "rb"))

# ================================
# FEATURE ENGINEERING (same as training)
# ================================
def bmi_category(bmi):
    if bmi < 18.5:
        return "underweight"
    elif bmi < 25:
        return "normal"
    elif bmi < 30:
        return "overweight"
    else:
        return "obese"

# ================================
# PREDICTION FUNCTION
# ================================
def predict_insurance(age, sex, bmi, children, smoker, region):

    # create bmi category
    bmi_cat = bmi_category(bmi)

    # create dataframe (IMPORTANT: same structure as training X)
    input_data = pd.DataFrame([[
        age,
        sex,
        bmi,
        children,
        smoker,
        region,
        bmi_cat
    ]], columns=[
        "age",
        "sex",
        "bmi",
        "children",
        "smoker",
        "region",
        "bmi_category"
    ])

    # prediction
    prediction = model.predict(input_data)[0]

    return f"Predicted Insurance Cost: ${prediction:.2f}"

# ================================
# GRADIO UI
# ================================
app = gr.Interface(
    fn=predict_insurance,
    inputs=[
        gr.Number(label="Age"),
        gr.Radio(["male", "female"], label="Sex"),
        gr.Number(label="BMI"),
        gr.Number(label="Children"),
        gr.Radio(["yes", "no"], label="Smoker"),
        gr.Radio(["southwest", "southeast", "northwest", "northeast"], label="Region")
    ],
    outputs="text",
    title="Insurance Cost Prediction System",
    description="Predict medical insurance cost using ML model (Random Forest + GridSearch)",
    live=False 
)

# ================================
# RUN APP
# ================================
if __name__ == "__main__":
    app.launch()