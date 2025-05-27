import streamlit as st
import joblib
import os
from googletrans import Translator

# ธีมสีพาสเทล ปรับขนาดและจัดตำแหน่งหน้าเว็บ
st.markdown("""
    <style>
    .main {
        max-width: 800px;
        margin: auto;
        background-color: #fefefe;
        padding: 2rem;
        border-radius: 12px;
    }
    .stTextArea textarea {
        background-color: #fff0f5;
        border-radius: 10px;
        border: 1px solid #e4bad4;
        font-size: 16px;
    }
    .stButton button {
        background-color: #c1f0dc;
        color: #4b2354;
        border-radius: 10px;
        font-weight: bold;
        font-size: 16px;
        padding: 0.5rem 1.5rem;
    }
    .stSuccess, .stWarning, .stInfo {
        background-color: #f1ebf9;
        border-left: 6px solid #b083d9;
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# โหลดโมเดลและเวกเตอร์
model = joblib.load(r"E:\!\B\complaint_model.pkl")
vectorizer = joblib.load(r"E:\!\B\tfidf_vectorizer.pkl")

# หัวข้อหน้าเว็บ
st.markdown("<h2 style='font-size:26px;'>🎨 Complaint Classification Web App</h2>", unsafe_allow_html=True)
st.markdown("""
<p style='font-size:18px;'>
พิมพ์ข้อความร้องเรียนด้านล่าง เพื่อจัดหมวดหมู่ให้อัตโนมัติ💬</p>
""", unsafe_allow_html=True)

# กล่องรับข้อความ
user_input = st.text_area("📝 พิมพ์ข้อความร้องเรียนที่นี่")

# เมื่อกดปุ่ม Predict
if st.button("🔍 Predict หมวดหมู่"):
    if user_input.strip() == "":
        st.warning("กรุณากรอกข้อความก่อน")
    else:
        # ถ้าเป็นภาษาไทย ให้แปลเป็นอังกฤษ
        if any('\u0E00' <= c <= '\u0E7F' for c in user_input):
            translator = Translator()
            translated = translator.translate(user_input, src='th', dest='en')
            user_input = translated.text
            st.info(f"🔄 แปลเป็นอังกฤษ: {user_input}")

        # แปลงข้อความและทำนาย
        vect_input = vectorizer.transform([user_input])
        prediction = model.predict(vect_input)

        # แปลผลลัพธ์เป็นภาษาไทย
        pred_en = prediction[0]
        product_thai = {
            "Debt collection": "ติดตามหนี้",
            "Mortgage": "สินเชื่อบ้าน",
            "Credit reporting, credit repair services, or other personal consumer reports": "รายงานเครดิต / ซ่อมแซมเครดิต",
            "Credit card or prepaid card": "บัตรเครดิต / พรีเพด",
            "Checking or savings account": "บัญชีเงินฝาก",
            "Money transfer, virtual currency, or money service": "โอนเงิน / สกุลเงินดิจิทัล",
            "Vehicle loan or lease": "สินเชื่อรถยนต์",
            "Payday loan, title loan, or personal loan": "สินเชื่อเงินด่วน / ส่วนบุคคล",
            "Student loan": "สินเชื่อนักเรียน",
            "Consumer loan": "สินเชื่อผู้บริโภค",
            "Bank account or service": "บัญชีธนาคาร / บริการ",
            "Money transfers": "โอนเงิน",
            "Prepaid card": "บัตรเติมเงิน",
            "Credit card": "บัตรเครดิต",
            "Virtual currency": "สกุลเงินดิจิทัล",
            "Other financial service": "บริการการเงินอื่น ๆ",
        }
        pred_th = product_thai.get(pred_en, "ไม่พบคำแปล")

        # เก็บค่าล่าสุดไว้ใช้กับ feedback
        st.session_state["last_input"] = user_input
        st.session_state["last_prediction"] = pred_en

        # แสดงผลลัพธ์
        st.success(f"📂 หมวดหมู่ที่ AI ทำนาย: {pred_en} ({pred_th})")

# ส่วนให้ feedback
if "last_prediction" in st.session_state and "last_input" in st.session_state:
    st.markdown("### 🙋‍♀️ คำตอบนี้ถูกต้องไหม?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ ถูกต้อง"):
            os.makedirs("Data", exist_ok=True)
            with open("Data/feedback.csv", "a", encoding="utf-8") as f:
                f.write(f"\"{st.session_state['last_input']}\",\"{st.session_state['last_prediction']}\",correct\n")
            st.success("ขอบคุณสำหรับคำติชม")
    with col2:
        if st.button("❌ ไม่ถูกต้อง"):
            os.makedirs("Data", exist_ok=True)
            with open("Data/feedback.csv", "a", encoding="utf-8") as f:
                f.write(f"\"{st.session_state['last_input']}\",\"{st.session_state['last_prediction']}\",incorrect\n")
            st.warning("ระบบจะนำไปปรับปรุงต่อไป")

# แสดงผลการประเมินโมเดล
if st.checkbox("📊 แสดงผลประเมินโมเดล"):
    try:
        with open("Model/metrics.txt", "r", encoding="utf-8") as f:
            st.text(f.read())
    except FileNotFoundError:
        st.warning("ยังไม่มีไฟล์ metrics.txt โปรด train โมเดลก่อน")
