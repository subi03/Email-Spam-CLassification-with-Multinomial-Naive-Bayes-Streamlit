import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


st.title('Email Spam Classifier')

# Load 
def load_data():
    data = pd.read_csv('spam.csv')
    data.columns = ['Category', 'Message']
    st.write(data.columns) 
    return data
df = load_data()

# Preprocessing the data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Message'])
y = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)  

# Splitting Train,Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Input 
user_input = st.text_area("Enter email text to classify", "")

if user_input:
    user_input_vector = vectorizer.transform([user_input])
    prediction = model.predict(user_input_vector)[0]

    if prediction == 1:
        st.write("### The email is classified as **SPAM**.")
    else:
        st.write("### The email is classified as **NOT SPAM**.")

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix
if st.button('Show Confusion Matrix'):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
    st.pyplot(fig)
    
