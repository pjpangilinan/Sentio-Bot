import pandas as pd
import re
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# --- Step 1: Data Collection & Balancing ---
print("Step 1: Loading and Preparing Data...")
# Load the full dataset (ensure 'Reviews.csv' is in the same directory)
# For demonstration, we'll create a dummy dataframe if the file is not found
try:
    df = pd.read_csv('reviews.csv')
except FileNotFoundError:
    print("Warning: 'Reviews.csv' not found.")

def map_sentiment(score):
    if score <= 2:
        return 'negative'
    elif score == 3:
        return 'neutral'
    else: # Scores 4 and 5
        return 'positive'

df['Sentiment'] = df['Score'].apply(map_sentiment)

print("\nCreating a smaller, balanced dataset of ~10,000 reviews...")
df_positive = df[df['Sentiment'] == 'positive']
df_negative = df[df['Sentiment'] == 'negative']
df_neutral = df[df['Sentiment'] == 'neutral']

# Ensure we don't try to sample more data than exists
min_class_size = min(len(df_positive), len(df_negative), len(df_neutral))
sample_size = min(3333, min_class_size)

df_positive_downsampled = df_positive.sample(n=sample_size, random_state=42)
df_negative_downsampled = df_negative.sample(n=sample_size, random_state=42)
df_neutral_downsampled = df_neutral.sample(n=sample_size, random_state=42)

df_balanced = pd.concat([df_positive_downsampled, df_negative_downsampled, df_neutral_downsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nWorking with a balanced sample of {len(df_balanced)} reviews.")
print("New balanced sentiment distribution:")
print(df_balanced['Sentiment'].value_counts())

# --- Step 2: Text Preprocessing ---
print("\nStep 2: Preprocessing Text Data...")
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    print("NLTK stopwords not found. Downloading...")
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("spaCy model 'en_core_web_sm' not found.")
    print("Please run 'python -m spacy download en_core_web_sm' in your terminal.")
    exit()

def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    doc = nlp(text.lower())
    processed_tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and token.lemma_ not in stop_words
    ]
    return ' '.join(processed_tokens)

df_balanced['Cleaned_Text'] = df_balanced['Text'].apply(preprocess_text)
print("Text preprocessing complete.")

# --- Step 3: Feature Engineering & Model Training ---
print("\nStep 3: Building and Training the Model Pipeline...")
X = df_balanced['Cleaned_Text']
y = df_balanced['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

# Using LogisticRegression which provides .predict_proba() for confidence scores
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
    ('clf', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
])

model_pipeline.fit(X_train, y_train)

# --- Step 4: Model Evaluation ---
print("\nStep 4: Evaluating the Model...")
y_pred = model_pipeline.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# --- Step 5: Save the Pipeline ---
print("\nStep 5: Saving the model pipeline...")
joblib.dump(model_pipeline, 'sentiment_model_pipeline.pkl')
print("Model pipeline saved to 'sentiment_model_pipeline.pkl'")

print("\n\n✅ NLP pipeline training and model saving complete.")
print("You can now run the Streamlit app using: streamlit run app.py")

