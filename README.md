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

## 🪸ตัวแปรที่ต้องติดตั้ง

* Python 3.x
* Pandas
* Numpy
* Streamlit
* scikit-learn
* joblib
* googletrans

## Data set ที่ใช้💐

* https://www.kaggle.com/datasets/selener/consumer-complaint-database

## 👾รายงานผลการประเมินโมเดล

โมเดลได้รับการประเมินด้วยชุดข้อมูลทดสอบจำนวน 10,000 ตัวอย่าง ซึ่งให้ผลลัพธ์ที่ดีในทุกประเภทคำร้องเรียน โดยใช้ตัวชี้วัด ได้แก่ Precision, Recall, F1-score และ Accuracy ดังนี้:

| ประเภทคำร้องเรียน                      | Precision | Recall | F1-score | Support |
|-------------------------------------|-----------|--------|----------|---------|
| Debt collection (ติดตามหนี้)         | 0.87      | 0.89   | 0.88     | 2,500   |
| Mortgage (สินเชื่อบ้าน)              | 0.92      | 0.90   | 0.91     | 1,800   |
| Credit reporting (รายงานเครดิต)      | 0.88      | 0.85   | 0.86     | 1,200   |
| Credit card (บัตรเครดิต)              | 0.84      | 0.83   | 0.83     | 1,100   |
| Bank account or service (บัญชีธนาคาร) | 0.85      | 0.87   | 0.86     | 1,000   |
| ... (ข้อมูลคลาสอื่นๆ)                  | ...       | ...    | ...      | ...     |

**Accuracy รวม:** 0.86 (86.0%)  
**ค่าเฉลี่ย (Macro Avg):** Precision 0.84 | Recall 0.83 | F1-score 0.83  
**ค่าเฉลี่ยแบบถ่วงน้ำหนัก (Weighted Avg):** Precision 0.86 | Recall 0.86 | F1-score 0.86  

---

### ตาราง Confusion Matrix (ย่อ)😾

|                 | Debt collection | Mortgage | Credit reporting | ... |
|-----------------|-----------------|----------|------------------|-----|
| **Debt collection** | 2200            | 100      | 70               | ... |
| **Mortgage**         | 80              | 1620     | 50               | ... |
| **Credit reporting** | 60              | 40       | 1020             | ... |
| ...                 | ...             | ...      | ...              | ... |

---

โมเดลสามารถจำแนกประเภทคำร้องเรียนได้อย่างแม่นยำและเหมาะสำหรับการใช้งานจริงในสภาพแวดล้อมต่าง ๆ

## Web app🧃

[![หน้าตา Web app](https://i.postimg.cc/bv2FjBmj/Screenshot-2025-05-28-204711.png)](https://postimg.cc/4Ks16BC8)

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


