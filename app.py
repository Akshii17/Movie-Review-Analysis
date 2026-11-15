import streamlit as st
import joblib
import re

st.set_page_config(page_title="Movie Review Sentiment Analysis", page_icon="🎬", layout="centered")

# ---------- Styling (background, colors, fonts) ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

/* Font */
html, body, .stApp,
[data-testid="stMarkdownContainer"] p,
[data-testid="stWidgetLabel"] p,
[data-testid="stCaptionContainer"],
.stButton button,
textarea,
div[data-baseweb="select"] div,
table {
    font-family: 'Poppins', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #fdf2f8 0%, #e0e7ff 50%, #dbeafe 100%);
}

/* Text colour */
.stApp h1, .stApp label, .stApp p, .stApp .stCaption {
    color: #1f2937 !important;
}

/* Bigger text everywhere */
[data-testid="stMarkdownContainer"] p {
    font-size: 0.85rem;
}
[data-testid="stWidgetLabel"] p {
    font-size: 0.85rem;
    font-weight: 600;
}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    font-size: 0.85rem;
}
div[data-baseweb="select"] div {
    font-size: 0.85rem;
}
textarea {
    font-size: 0.85rem !important;
}
.stButton button p {
    font-size: 0.85rem;
    font-weight: 500;
}
table {
    font-size: 0.85rem;
}

/* Title */
.stApp p.title {
    text-align: center;
    font-size: 2rem;
    font-weight: 700;
    color: #312e81 !important;
    margin-bottom: 0;
}
.stApp p.subtitle {
    text-align: center;
    font-size: 0.85rem;
    color: #4b5563 !important;
    margin-bottom: 1.5rem;
}

/* Result tag */
.result {
    padding: 0.9rem 1.1rem;
    border-radius: 10px;
    font-family: 'Poppins', sans-serif;
    font-size: 1.2rem;
    font-weight: 600;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
    display: block;
}
</style>
""", unsafe_allow_html=True)

# ---------- Load models ----------
vec = joblib.load("tfidf_vectorizer.joblib")
models = joblib.load("all_models.joblib")
accs = joblib.load("accuracies.joblib")


def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)
    return text.lower()


def predict(model, text):
    X = vec.transform([clean_text(text)])
    pred = model.predict(X)[0]
    conf = None
    if hasattr(model, "predict_proba"):
        conf = model.predict_proba(X)[0][pred]
    return pred, conf


def set_text(t):
    st.session_state.review = t


# ---------- Header ----------
st.markdown('<p class="title">🎬 Movie Review Sentiment Analysis</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Type a movie review and find out if it is positive or negative</p>',
            unsafe_allow_html=True)

# ---------- Model selection ----------
choice = st.selectbox("Choose a model", list(models.keys()))
st.caption(f"Test accuracy of {choice}: {accs[choice] * 100:.2f}%")

# ---------- Example buttons ----------
c1, c2 = st.columns(2)
c1.button("😊 Try a positive example", on_click=set_text,
          args=("This movie was amazing! Great acting and a beautiful story.",),
          use_container_width=True)
c2.button("😞 Try a negative example", on_click=set_text,
          args=("Terrible film. Boring plot and very bad acting.",),
          use_container_width=True)

# ---------- Input ----------
review = st.text_area("Enter a movie review", key="review", height=150)

# ---------- Prediction ----------
if st.button("Predict", type="primary", use_container_width=True):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        pred, conf = predict(models[choice], review)

        if pred == 1:
            label, color, bg = "Positive", "#16a34a", "#dcfce7"
        else:
            label, color, bg = "Negative", "#dc2626", "#fee2e2"

        st.markdown(
            f'<div class="result" style="background:{bg}; border-left:6px solid {color}; color:{color};">'
            f'{label}</div>',
            unsafe_allow_html=True,
        )

        if conf is not None:
            st.write(f"Confidence: **{conf * 100:.1f}%**")
            st.progress(float(conf))

        with st.expander("See what all models say"):
            rows = {"Model": [], "Prediction": [], "Accuracy": []}
            for name, m in models.items():
                p, _ = predict(m, review)
                rows["Model"].append(name)
                rows["Prediction"].append("Positive" if p == 1 else "Negative")
                rows["Accuracy"].append(f"{accs[name] * 100:.2f}%")
            st.table(rows)