# 🌼เว็บแอปจำแนกประเภทคำร้องเรียน

โปรเจกต์นี้เป็นเว็บแอปพลิเคชันสำหรับจำแนกประเภทของข้อความคำร้องเรียน (Complaint) โดยใช้โมเดลแมชชีนเลิร์นนิง ซึ่งช่วยทำนายว่าคำร้องเรียนนั้นเกี่ยวข้องกับหมวดหมู่สินค้าหรือบริการทางการเงินประเภทใดในจำนวน 16 ประเภทที่กำหนดไว้

## จุดประสงค์ของงานนี้🌻

เพื่อพัฒนาเว็บแอปพลิเคชันที่ใช้โมเดลแมชชีนเลิร์นนิงในการจำแนกประเภทของข้อความคำร้องเรียนให้เป็นหมวดหมู่สินค้าหรือบริการทางการเงินที่กำหนดไว้ล่วงหน้า (ทั้งหมด 16 ประเภท) เพื่อช่วยให้การวิเคราะห์และจัดการคำร้องเรียนเป็นไปอย่างรวดเร็วและแม่นยำมากขึ้น

โดยระบบจะรับข้อความคำร้องเรียนจากผู้ใช้ วิเคราะห์และทำนายว่าคำร้องเรียนนั้นเกี่ยวข้องกับบริการทางการเงินประเภทใด ช่วยผู้ดูแลระบบหรือองค์กรในการจัดการข้อมูลคำร้องเรียนอย่างมีประสิทธิภาพ

## 🎏หมวดหมู่สินค้า (14 คลาส)

1. การรายงานสินเชื่อ (Credit reporting)  
2. การจัดเก็บหนี้ (Debt collection)  
3. สินเชื่อที่อยู่อาศัย (Mortgage)  
4. บัตรเครดิต (Credit card)  
5. บัญชีธนาคารหรือบริการ (Bank account or service)  
6. สินเชื่อเพื่อผู้บริโภค (Consumer loan)  
7. บัญชีเงินฝากหรือออมทรัพย์ (Checking or savings account)  
8. สินเชื่อนักศึกษา (Student loan)  
9. สินเชื่อเงินด่วน (Payday loan)  
10. สินเชื่อหรือเช่าซื้อรถยนต์ (Vehicle loan or lease)  
11. การโอนเงิน (Money transfers)  
12. บัตรเติมเงิน (Prepaid card)  
13. สกุลเงินเสมือน (Virtual currency)  
14. บริการทางการเงินอื่นๆ (Other financial service)  


## วิธีใช้งาน💖

1. โคลนโปรเจกต์นี้:

```bash
git clone https://github.com/yourusername/complaint-classification.git
cd complaint-classification
````

2. ติดตั้ง dependencies:

```bash
pip install -r requirements.txt
```

3. วางไฟล์โมเดล (`complaint_model.pkl`) และตัวแปลงข้อความ (`tfidf_vectorizer.pkl`) ในโฟลเดอร์โปรเจกต์ หรือตั้งค่า path ให้ถูกต้องในไฟล์ `app.py`

4. รันแอปด้วยคำสั่ง:

```bash
streamlit run app.py
```

5. เปิดเบราว์เซอร์ที่ URL ที่แสดง (ปกติจะเป็น [http://localhost:8501](http://localhost:8501))

## 🟢การใช้งาน

* กรอกข้อความคำร้องเรียนในช่องข้อความ
* กดปุ่ม **Predict**
* ระบบจะแสดงผลการทำนายประเภทสินค้าหรือบริการที่เกี่ยวข้อง

## โครงสร้างโปรเจกต์🍒

* `app.py` - ไฟล์เว็บแอปหลักที่ใช้ Streamlit
* `complaint_model.pkl` - โมเดลแมชชีนเลิร์นนิงที่ฝึกไว้แล้ว
* `tfidf_vectorizer.pkl` - ตัวแปลงข้อความ TF-IDF

## 🪸ตัวแปรที่ต้องติดตั้ง

* Python 3.x
* Streamlit
* scikit-learn
* joblib

## Data set ที่ใช้💐


* https://www.kaggle.com/datasets/selener/consumer-complaint-database

## 🌤 โครงสร้างโปรเจกต์


complaint-classification/

├── Data/ # เก็บข้อมูล Feedback ที่ผู้ใช้ส่งกลับมา

│ └── feedback.csv # ไฟล์เก็บผลตอบรับถูก/ผิดจากผู้ใช้

├── Model/ # เก็บไฟล์โมเดลและผลประเมิน

│ ├── complaint_model.pkl # โมเดล ML ที่เทรนแล้ว

│ ├── metrics.txt # รายงานผลการประเมินโมเดล

│ └── train.py # สคริปต์เทรนโมเดล

├── venv/ # ไฟล์และโฟลเดอร์ของ virtual environment (Python)

├── app.py # โค้ดเว็บแอป Streamlit สำหรับทำนายและรับ feedback

├── complaint_model.pkl # (สำรอง) โมเดล ML สำหรับใช้งาน

├── README.md # คำอธิบายโปรเจกต์และวิธีใช้งาน

├── requirements.txt # ไลบรารีที่ต้องติดตั้งสำหรับโปรเจกต์

├── rows.csv # Dataset ดิบสำหรับเทรนโมเดล (ข้อมูลคำร้องเรียน)

└── tfidf_vectorizer.pkl # ตัวแปลงข้อความเป็นเวกเตอร์สำหรับโมเดล

## 🔗 แหล่งข้อมูลอ้างอิง


* [Consumer Complaint Database - U.S. Consumer Financial Protection Bureau (CFPB)]

https://www.consumerfinance.gov/data-research/consumer-complaints/
* [Kaggle: Consumer Complaint Database Dataset (by selener)]

https://www.kaggle.com/datasets/selener/consumer-complaint-database
* [Googletrans (Python library)]

https://github.com/ssut/py-googletrans
* [Streamlit Documentation]

https://docs.streamlit.io/
* [scikit-learn: Machine Learning in Python]

https://scikit-learn.org/


