
import re
from io import BytesIO

import nltk
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

st.set_page_config(
    page_title="SmartDoc - Document Summarizer",
    page_icon="📄",
    layout="wide"
)


def extract_text(pdf_file):
    reader = PdfReader(BytesIO(pdf_file.getvalue()))
    pages = []

    for page in reader.pages:
        pages.append(page.extract_text() or "")

    return "\n".join(pages), len(reader.pages)


def preprocess_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?;:%()-]", " ", text)
    return text.strip()



def generate_summary(text, summary_percentage):
    sentences = nltk.sent_tokenize(text)

    if not sentences:
        return "", [], []

    if len(sentences) <= 2:
        return " ".join(sentences), [], sentences

    vectorizer = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return " ".join(sentences), [], sentences

    # Calculate sentence scores as a standard NumPy array
    sentence_scores = tfidf_matrix.sum(axis=1)
    sentence_scores = (
        sentence_scores.A.ravel()
        if hasattr(sentence_scores, "A")
        else sentence_scores.ravel()
    )

    nonzero_counts = tfidf_matrix.getnnz(axis=1).ravel()
    sentence_scores = sentence_scores / (nonzero_counts + 1e-9)

    # Rank sentence indices
    ranked_indices = sentence_scores.argsort()[::-1]

    count = max(
        1,
        min(
            len(sentences),
            int(round(len(sentences) * summary_percentage / 100))
        )
    )

    # Convert NumPy indices into Python integers
    selected_indices = sorted(
        int(index) for index in ranked_indices[:count]
    )

    summary = " ".join(
        sentences[index] for index in selected_indices
    )

    # Calculate keyword importance scores
    keyword_scores = tfidf_matrix.sum(axis=0)
    keyword_scores = (
        keyword_scores.A.ravel()
        if hasattr(keyword_scores, "A")
        else keyword_scores.ravel()
    )

    terms = vectorizer.get_feature_names_out()
    top_indices = keyword_scores.argsort()[::-1][:10]

    keywords = [
        str(terms[int(index)])
        for index in top_indices
        if keyword_scores[int(index)] > 0
    ]

    return summary, keywords, sentences



def word_count(text):
    return len(re.findall(r"\b\w+\b", text))


st.title("SmartDoc: Document Summarization System")

st.write(
    "An Information Retrieval application that generates "
    "extractive summaries using TF-IDF."
)

st.sidebar.header("Summary Settings")

summary_percentage = st.sidebar.slider(
    "Summary length (%)",
    min_value=10,
    max_value=80,
    value=30,
    step=10
)

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)

if uploaded_file is not None:
    try:
        raw_text, total_pages = extract_text(uploaded_file)
        cleaned_text = preprocess_text(raw_text)

        if not cleaned_text:
            st.error(
                "No readable text was found. Please upload a "
                "text-based PDF document."
            )

        else:
            st.success(
                "PDF uploaded and text extracted successfully."
            )

            st.subheader("Document Statistics")

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Pages", total_pages)
            col2.metric("Total Words", word_count(cleaned_text))

            total_sentences = len(nltk.sent_tokenize(cleaned_text))
            col3.metric("Total Sentences", total_sentences)

            if st.button("Generate Summary", type="primary"):
                summary, keywords, sentences = generate_summary(
                    cleaned_text,
                    summary_percentage
                )

                st.subheader("Generated Summary")
                st.write(summary)

                original_words = word_count(cleaned_text)
                summary_words = word_count(summary)

                compression = (
                    (1 - summary_words / original_words) * 100
                    if original_words > 0
                    else 0
                )

                col1, col2, col3 = st.columns(3)

                col1.metric("Summary Words", summary_words)
                col2.metric(
                    "Compression",
                    f"{max(0, compression):.2f}%"
                )
                col3.metric(
                    "Summary Setting",
                    f"{summary_percentage}%"
                )

                st.subheader("Important Keywords")

                if keywords:
                    st.write(" • ".join(keywords))
                else:
                    st.write("No keywords could be extracted.")

                st.download_button(
                    label="Download Summary",
                    data=summary,
                    file_name="smartdoc_summary.txt",
                    mime="text/plain"
                )

    except Exception as error:
        st.error(f"Unable to process the PDF: {error}")

else:
    st.info("Upload a PDF document to begin summarization.")

st.divider()
st.caption("SmartDoc | Information Retrieval Mini Project")
