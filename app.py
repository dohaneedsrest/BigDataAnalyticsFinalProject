# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


@st.cache_data
def load_data():
    df = pd.read_csv("Exam_Score_Prediction.csv")
    return df

@st.cache_resource
def load_model_objects():
    with open("model_objects.pkl", "rb") as f:
        objects = pickle.load(f)
    return objects

df = load_data()
objects = load_model_objects()

model = objects['model']
scaler = objects['scaler']
feature_order = objects['feature_order']


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Visualized EDA", "Prediction"])


if page == "Home":
    st.title("Student Exam Score Analysis Project")
    st.markdown("""
    **Overview:** This project predicts student exam scores based on demographic, academic, and lifestyle features.
    
    **Features used:**
    - Age, Gender, Course, Study Hours, Attendance, Internet Access
    - Sleep Hours, Sleep Quality, Study Method, Facility Rating, Exam Difficulty
    """)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    st.write("Dataset Shape:", df.shape)


elif page == "Visualized EDA":
    st.title("Visualized Exploratory Data Analysis (EDA)")

    # Histogram of exam scores
    st.subheader("Distribution of Exam Scores")
    fig, ax = plt.subplots()
    ax.hist(df['exam_score'], bins=30, color='skyblue', edgecolor='black')
    ax.set_xlabel('Exam Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Exam Scores')
    st.pyplot(fig)

    # Bar plots for categorical features
    st.subheader("Categorical Feature Distributions")
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    for col in categorical_cols:
        st.markdown(f"**{col}**")
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind='bar', color='lightgreen', ax=ax)
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of {col}')
        st.pyplot(fig)


    # Scatter plots
    st.subheader("Scatter Plots: Study Hours & Attendance vs Exam Score")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(df['study_hours'], df['exam_score'], color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Study Hours')
    axes[0].set_ylabel('Exam Score')
    axes[0].set_title('Study Hours vs Exam Score')
    axes[0].grid(alpha=0.3)

    axes[1].scatter(df['class_attendance'], df['exam_score'], color='lightcoral', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Class Attendance')
    axes[1].set_ylabel('Exam Score')
    axes[1].set_title('Attendance vs Exam Score')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Box plot: Exam Score by Study Method
    st.subheader("Box Plot: Exam Score by Study Method")
    fig, ax = plt.subplots()
    df.boxplot(column='exam_score', by='study_method', ax=ax)
    ax.set_title('Exam Score by Study Method')
    ax.set_xlabel('Study Method')
    ax.set_ylabel('Exam Score')
    plt.suptitle('')
    st.pyplot(fig)

    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[numerical_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)
    
elif page == "Prediction":
    st.title("Predict Exam Score")
    st.write("Enter student details to predict exam score:")

    age = st.slider("Age", int(df.age.min()), int(df.age.max()), int(df.age.mean()))
    study_hours = st.slider("Study Hours", float(df.study_hours.min()), float(df.study_hours.max()), float(df.study_hours.mean()))
    class_attendance = st.slider("Class Attendance (%)", float(df.class_attendance.min()), float(df.class_attendance.max()), float(df.class_attendance.mean()))
    sleep_hours = st.slider("Sleep Hours", float(df.sleep_hours.min()), float(df.sleep_hours.max()), float(df.sleep_hours.mean()))

    gender = st.selectbox("Gender", df['gender'].unique())
    course = st.selectbox("Course", df['course'].unique())
    internet_access = st.selectbox("Internet Access", df['internet_access'].unique())
    sleep_quality = st.selectbox("Sleep Quality", df['sleep_quality'].unique())
    study_method = st.selectbox("Study Method", df['study_method'].unique())
    facility_rating = st.selectbox("Facility Rating", df['facility_rating'].unique())
    exam_difficulty = st.selectbox("Exam Difficulty", df['exam_difficulty'].unique())

    input_dict = {
        'age': age,
        'study_hours': study_hours,
        'class_attendance': class_attendance,
        'sleep_hours': sleep_hours,
        'gender': gender,
        'course': course,
        'internet_access': internet_access,
        'sleep_quality': sleep_quality,
        'study_method': study_method,
        'facility_rating': facility_rating,
        'exam_difficulty': exam_difficulty
    }
    input_df = pd.DataFrame([input_dict])

    cat_cols = ['gender', 'course', 'internet_access', 'sleep_quality',
                'study_method', 'facility_rating', 'exam_difficulty']
    input_df = pd.get_dummies(input_df, columns=cat_cols)

    for col in feature_order:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_order]

    input_scaled = scaler.transform(input_df)

    if st.button("Predict Exam Score"):
        prediction = model.predict(input_scaled)
        st.success(f"Predicted Exam Score: {prediction[0]:.2f}")
