import streamlit as st
import joblib

# กำหนดพาธไฟล์โมเดลและ vectorizer แบบเต็ม (raw string)
model_path = r"E:\!\B\complaint_model.pkl"
vectorizer_path = r"E:\!\B\tfidf_vectorizer.pkl"

# โหลดโมเดลและ vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

st.title("Complaint Classification Web App")

user_input = st.text_area("Enter complaint text here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some complaint text.")
    else:
        vect_input = vectorizer.transform([user_input])
        prediction = model.predict(vect_input)
        st.success(f"Predicted Product Category: {prediction[0]}")