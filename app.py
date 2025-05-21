import streamlit as st  # นำเข้าไลบรารี Streamlit สำหรับสร้างเว็บแอปง่าย ๆ
import joblib           # นำเข้าไลบรารี joblib สำหรับโหลดโมเดลและ vectorizer ที่เซฟไว้

# กำหนดพาธไฟล์โมเดลและ vectorizer แบบเต็ม (raw string) เพื่อหลีกเลี่ยงปัญหาเรื่องเครื่องหมายพิเศษในพาธไฟล์
model_path = r"E:\!\B\complaint_model.pkl"        # พาธไฟล์โมเดลที่ฝึกไว้แล้ว
vectorizer_path = r"E:\!\B\tfidf_vectorizer.pkl"  # พาธไฟล์ TF-IDF vectorizer ที่ใช้แปลงข้อความ

# โหลดโมเดลและ vectorizer จากไฟล์ที่กำหนด
model = joblib.load(model_path)         # โหลดโมเดล machine learning
vectorizer = joblib.load(vectorizer_path)  # โหลดตัวแปลงข้อความ TF-IDF

# กำหนดชื่อหัวข้อของเว็บแอป
st.title("Complaint Classification Web App")

# สร้างกล่องข้อความให้ผู้ใช้ป้อนข้อความคำร้องเรียน
user_input = st.text_area("Enter complaint text here:")

# สร้างปุ่ม Predict เมื่อกดจะเริ่มทำนาย
if st.button("Predict"):
    # ตรวจสอบว่าผู้ใช้ป้อนข้อความหรือไม่ ถ้าไม่ป้อนจะเตือนให้ป้อนข้อความก่อน
    if user_input.strip() == "":
        st.warning("Please enter some complaint text.")
    else:
        # แปลงข้อความที่ผู้ใช้ป้อนเป็นเวกเตอร์โดยใช้ vectorizer
        vect_input = vectorizer.transform([user_input])
        # ใช้โมเดลทำนายหมวดหมู่จากเวกเตอร์ข้อความ
        prediction = model.predict(vect_input)
        # แสดงผลลัพธ์การทำนายบนเว็บ
        st.success(f"Predicted Product Category: {prediction[0]}")
