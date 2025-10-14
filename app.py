import streamlit as st
import spacy
import joblib
import re
import nltk
from nltk.corpus import stopwords
import subprocess

st.set_page_config(page_title="Sentio-Bot", layout="centered")

# --- Error Handling for NLTK and Model Loading ---
try:
    stopwords.words('english')
except LookupError:
    st.info("Downloading NLTK stopwords...")
    nltk.download('stopwords')
    
@st.cache_resource
def load_models():
    """Load the model pipeline and spaCy model."""
    try:
        # Load the single pipeline object that includes vectorizer and classifier
        model_pipeline = joblib.load('sentiment_model_pipeline.pkl')
        nlp = spacy.load('en_core_web_sm')
        return {
            "pipeline": model_pipeline,
            "nlp": nlp
        }
    except FileNotFoundError:
        st.error(
            "Model file ('sentiment_model_pipeline.pkl') not found. "
            "Please run the nlp_pipeline.py script to train and save the model."
        )
        st.stop()
    except OSError:
        st.error(
            "spaCy model 'en_core_web_sm' not found. "
            "Downloading...."
        )
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm", "--user"])
        nlp = spacy.load('en_core_web_sm')
        return { "pipeline": model_pipeline, "nlp": nlp }

# Load the models
models = load_models()
model_pipeline = models["pipeline"]
nlp = models["nlp"]

stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    """Preprocesses text for the model pipeline."""
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    doc = nlp(text.lower())
    processed_tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and token.lemma_ not in stop_words
    ]
    return ' '.join(processed_tokens)


def ner_to_html(text, doc):
    """Converts spaCy doc entities to a colored HTML string."""
    html_parts = []
    last_end = 0
    colors = {
        "PERSON": "#ffcce0", "ORG": "#cce0ff", "GPE": "#d4edda", "LOC": "#fff3cd",
        "PRODUCT": "#f8d7da", "EVENT": "#e2e3e5", "DATE": "#d1ecf1", "MONEY": "#d6d8db"
    }
    for ent in doc.ents:
        html_parts.append(text[last_end:ent.start_char])
        entity_type = ent.label_
        color = colors.get(entity_type, "#f0f2f5")
        html_parts.append(
            f'<span style="background-color: {color}; padding: 0.3em 0.5em; margin: 0 0.2em; line-height: 1; border-radius: 0.35em; border: 1px solid #ddd;">'
            f'{text[ent.start_char:ent.end_char]}'
            f'<strong style="font-size: 0.8em; font-weight: bold; margin-left: 0.5em; color: #333; border-left: 2px solid #aaa; padding-left: 0.4em;">{entity_type}</strong>'
            f'</span>'
        )
        last_end = ent.end_char
    html_parts.append(text[last_end:])
    return "".join(html_parts)
st.markdown(
    "<h2 style='text-align: center; font-family: Inter, sans-serif; margin-bottom: -2rem;'>Sentio-Bot</h2>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; font-family: Inter, sans-serif; margin-bottom: 0.5rem;'>Sentiment Analysis with Named Entity Recognition</p>",
    unsafe_allow_html=True
)

if "analysis_report" not in st.session_state:
    st.session_state.analysis_report = None
if "history" not in st.session_state:
    st.session_state.history = []

st.sidebar.title("History")
if st.session_state.history:
    for i, item in enumerate(reversed(st.session_state.history)):
        if st.sidebar.button(f"{item['text'][:40]}...", key=f"hist_{i}"):
            st.session_state.analysis_report = item
else:
    st.sidebar.info("No analyses yet. ")

prompt = st.chat_input("Analyze this text...")

if prompt:
    # Preprocess the text first
    preprocessed_prompt = preprocess_text(prompt)

    # Use the pipeline to predict and get probabilities
    prediction = model_pipeline.predict([preprocessed_prompt])[0]
    confidence = model_pipeline.predict_proba([preprocessed_prompt]).max()

    # Perform NER on the original text
    doc = nlp(prompt)

    st.session_state.analysis_report = {
        "text": prompt,
        "sentiment": prediction.capitalize(),
        "confidence": confidence,
        "doc": doc
    }
    st.session_state.history.append(st.session_state.analysis_report)

if st.session_state.analysis_report:
    report = st.session_state.analysis_report
    sentiment = report['sentiment']
    confidence = report['confidence']
    doc = report['doc']
    text = report['text']

    sentiment_bg_colors = {"Positive": "#d4edda", "Negative": "#f8d7da", "Neutral": "#e2e3e5"}
    sentiment_text_colors = {"Positive": "#155724", "Negative": "#721c24", "Neutral": "#383d41"}
    bg_color = sentiment_bg_colors.get(sentiment, "#f0f2f5")
    text_color = sentiment_text_colors.get(sentiment, "#333")

    ner_html = ner_to_html(text,
                           doc) if doc.ents else '<p style="font-style: italic; color: #666;">No named entities were found.</p>'

    card_html = f"""
    <div style="border: 1px solid #ddd; border-radius: 10px; padding: 25px; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1); margin-top: 20px;">
        <h4 style="margin-bottom: 20px; border-bottom: 1px solid #eee; padding-bottom: 10px;">Analysis Report</h4>
        <div style="display: flex; justify-content: space-around; align-items: center; margin-bottom: 25px;">
            <div style="text-align: center;">
                <p style="font-size: 1em; margin-bottom: 5px; color: #555;">Predicted Sentiment</p>
                <p style="background-color: {bg_color}; color: {text_color}; padding: 8px 15px; border-radius: 7px; font-weight: bold; font-size: 1.2em;">{sentiment}</p>
            </div>
            <div style="text-align: center;">
                <p style="font-size: 1em; margin-bottom: 5px; color: #555;">Confidence Score</p>
                <p style="font-weight: bold; font-size: 1.2em; color: #333; padding: 8px 15px;">{confidence:.2%}</p>
            </div>
        </div>
        <h5 style="margin-top: 20px; margin-bottom: 15px; border-bottom: 1px solid #eee; padding-bottom: 5px;">Named Entity Recognition (NER)</h5>
        <div style="line-height: 2.2;">{ner_html}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)











