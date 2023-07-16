import streamlit as st
from ntypesclassification.ipynb import train_model, classify_text

# Load the model
vectorizer, model, embeddings, cluster_labels, cluster_types = train_model('dataaaa.csv')

# Streamlit app
def main():
    st.title('Text Category Detection')

    # User input
    text_input = st.text_input('Enter a text sample:', '')

    # Classify user input
    if text_input:
        # Classify the text
        category_type = classify_text(text_input, vectorizer, model, embeddings, cluster_labels, cluster_types)

        # Display the detected category type
        st.write(f'Detected Category Type: {category_type}')

if __name__ == '__main__':
    main()
