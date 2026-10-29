import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import streamlit as st

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# Text preprocessing function
def clean_text(text):
  text = ''.join([c.lower() for c in text if c.isalpha() or c == ' '])
  words = text.split()
  return ' '.join([lemmatizer.lemmatize(w) for w in words if w not in stop_words])


# Load saved model & vectorizer with caching for fast performance
@st.cache_resource
def load_assets():
  model = joblib.load('spam_model.pkl')
  vectorizer = joblib.load('tfidf_vectorizer.pkl')
  return model, vectorizer


model, vectorizer = load_assets()

# --- Streamlit UI Layout ---
st.set_page_config(
    page_title='Spam & Phishing Detector', page_icon='🛡️', layout='centered'
)

st.title('🛡️ SMS Spam & Phishing Detector')
st.write(
    'Enter a message below to evaluate whether it is classified as **Spam**'
    ' or **Legitimate (Ham)**.'
)

# Text input widget
user_input = st.text_area(
    'Input SMS / Email Text:',
    height=120,
    placeholder=(
        'e.g., URGENT! You have won a $1,000 gift card. Call 0800-1234 to claim'
        ' now.'
    ),
)

# Sidebar settings and info
st.sidebar.header('Project Overview')
st.sidebar.markdown("""
- **Model:** Multinomial Naive Bayes
- **Feature Extractor:** TF-IDF Vectorizer
- **Module Project:** NLP Text Classification
""")

# Predict action
if st.button('Analyze Message', type='primary'):
  if user_input.strip() == '':
    st.warning('Please enter a message first.')
  else:
    # 1. Preprocess & Vectorize
    cleaned = clean_text(user_input)
    vec_input = vectorizer.transform([cleaned])

    # 2. Predict
    prediction = model.predict(vec_input)[0]
    probabilities = model.predict_proba(vec_input)[0]

    spam_prob = probabilities[1] * 100
    ham_prob = probabilities[0] * 100

    st.subheader('Analysis Results')

    # Display result alert box
    if prediction == 1:
      st.error(f'🚨 **Classification: SPAM** ({spam_prob:.1f}% Confidence)')
    else:
      st.success(f'✅ **Classification: LEGITIMATE** ({ham_prob:.1f}% Confidence)')

    # Display probability progress bar
    st.write('**Model Confidence Breakdown:**')
    st.progress(int(spam_prob))
    st.caption(
        f'Spam Probability: **{spam_prob:.2f}%** | Legitimate Probability:'
        f' **{ham_prob:.2f}%**'
    )