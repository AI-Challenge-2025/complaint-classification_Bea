import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# โหลดและเตรียมข้อมูล
df = pd.read_csv('rows.csv', low_memory=False)
df = df[['Consumer complaint narrative', 'Product']].dropna()
df['Consumer complaint narrative'] = df['Consumer complaint narrative'].str.lower()

# ตัดคลาสที่มีจำนวนน้อย
min_support = 100
value_counts = df['Product'].value_counts()
selected_classes = value_counts[value_counts >= min_support].index.tolist()
df = df[df['Product'].isin(selected_classes)]

print(f"Remaining classes after filtering: {len(selected_classes)}")
print("Classes:", selected_classes)

# แยกฟีเจอร์และเป้าหมาย
X = df['Consumer complaint narrative']
y = df['Product']

# แบ่งข้อมูล train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# สร้าง TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# สร้างและเทรนโมเดล
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# ประเมินโมเดล
y_pred = model.predict(X_test_tfidf)
report = classification_report(y_test, y_pred, zero_division=0)

# บันทึกผลประเมิน
with open('Model/metrics.txt', 'w') as f:
    f.write(report)

# เซฟโมเดลและ vectorizer
joblib.dump(model, 'complaint_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("Model trained and metrics.txt generated.")
