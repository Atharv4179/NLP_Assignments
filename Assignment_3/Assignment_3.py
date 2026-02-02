import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('stopwords')
nltk.download('wordnet')


documents = [
    "My Self Atharva",
    "My Profession Engineer",
    "I like Sports"
]

labels = ["Person", "Profession", "Hobby"]

df = pd.DataFrame({
    "Text": documents,
    "Label": labels
})

print("Original Data:")
print(df)
print("="*60)


def clean_text(text):
    text = text.lower()                    
    text = re.sub(r'[^a-z\s]', '', text)   
    return text

df["Cleaned_Text"] = df["Text"].apply(clean_text)


stop_words = set(stopwords.words("english"))

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stop_words])

df["No_Stopwords"] = df["Cleaned_Text"].apply(remove_stopwords)


lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

df["Lemmatized_Text"] = df["No_Stopwords"].apply(lemmatize_text)

print("After Cleaning, Stopword Removal & Lemmatization:")
print(df[["Text", "Lemmatized_Text"]])
print("="*60)


label_encoder = LabelEncoder()
df["Encoded_Label"] = label_encoder.fit_transform(df["Label"])

print("Label Encoding:")
print(df[["Label", "Encoded_Label"]])
print("="*60)


tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df["Lemmatized_Text"])

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)

print("TF-IDF Representation:")
print(tfidf_df)
print("="*60)


df.to_csv("processed_text_output.csv", index=False)
tfidf_df.to_csv("tfidf_output.csv", index=False)

print("Outputs saved successfully:")
print("1. processed_text_output.csv")
print("2. tfidf_output.csv")
