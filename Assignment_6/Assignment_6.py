# Import libraries
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

nltk.download('punkt')


documents = [
    "Artificial Intelligence is transforming sports analytics",
    "Machine Learning helps coaches analyze player performance",
    "India won the final match and players celebrated the victory",
    "Sports analytics uses data science and machine learning techniques"
]

print("Documents:\n")
for i, doc in enumerate(documents):
    print(f"D{i+1}: {doc}")



# a) Count Occurrence
count_vectorizer = CountVectorizer()
bow_counts = count_vectorizer.fit_transform(documents)

bow_df = pd.DataFrame(
    bow_counts.toarray(),
    columns=count_vectorizer.get_feature_names_out()
)

print("\nBag of Words - Count Occurrence:\n")
print(bow_df)

# b) Normalized Count Occurrence
bow_normalized = bow_counts.toarray() / bow_counts.toarray().sum(axis=1, keepdims=True)

bow_norm_df = pd.DataFrame(
    bow_normalized,
    columns=count_vectorizer.get_feature_names_out()
)

print("\nBag of Words - Normalized Count Occurrence:\n")
print(bow_norm_df)



tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)

print("\nTF-IDF Representation:\n")
print(tfidf_df)



# Tokenize sentences
tokenized_docs = [nltk.word_tokenize(doc.lower()) for doc in documents]

# Train Word2Vec model
w2v_model = Word2Vec(
    sentences=tokenized_docs,
    vector_size=50,
    window=3,
    min_count=1,
    workers=4
)

# Get word embeddings
print("\nWord2Vec Embedding for 'sports':\n")
print(w2v_model.wv['sports'])

print("\nSimilar words to 'analytics':\n")
print(w2v_model.wv.most_similar('analytics'))
