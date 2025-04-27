# trainmodel.py (REAL DATA VERSION)

import pandas as pd
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. Load your full dataset
df = pd.read_csv('UpdatedResumeDataSet.csv', encoding='utf-8')

# 2. Cleaning function (same cleaning like app.py)
def clean_text(text):
    text = re.sub('http\S+\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\S+', '', text)
    text = re.sub('@\S+', ' ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub('\s+', ' ', text)
    return text.strip()

# 3. Clean the Resume Texts
df['cleaned_resume'] = df['Resume'].apply(clean_text)

# 4. Define X and y
X = df['cleaned_resume']
y = df['Category']

# 5. Vectorize the Texts
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# 6. Train-Test Split (Optional, for validation)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42, stratify=y)

# 7. Train the Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8. (Optional) Print Accuracy
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"✅ Training Accuracy: {train_acc*100:.2f}%")
print(f"✅ Testing Accuracy: {test_acc*100:.2f}%")

# 9. Save the model and vectorizer
joblib.dump(model, 'resume_categorization_model.pkl')
joblib.dump(vectorizer, 'resume_vectorizer.pkl')

print("✅ Model and Vectorizer have been trained and saved successfully!")
