import streamlit as st
from model import predict_news, get_keywords

# Title
st.markdown("# 📰 AI Fake News Detection System")
st.markdown("### Analyze news using Machine Learning")

# Info box
st.info("Enter a detailed news article to get better prediction results.")

# Input box
user_input = st.text_area("Enter News Text")

# Button
if st.button("Predict"):

    # Validation
    if len(user_input.split()) < 10:
        st.warning("Please enter at least 10 words for better prediction")
    else:
        prediction, confidence = predict_news(user_input)

        # Result heading
        st.subheader("Result")

        # Color output
        if prediction == "Real News":
            st.success(prediction)
        else:
            st.error(prediction)

        # Confidence
        st.write("Confidence:", round(confidence, 2))

        # Progress bar
        st.progress(float(confidence))

        # Keywords
        keywords = get_keywords(user_input)
        st.write("Important Words:", keywords)