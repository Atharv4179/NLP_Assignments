import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

documents = [
    "My Self Atharva",
    "My Profession Engineer",
    "I like Sports"
]

print("Documents:")
for i, doc in enumerate(documents, 1):
    print(f"{i}. {doc}")

print("\n" + "="*60)

count_vectorizer = CountVectorizer(lowercase=True)
bow_counts = count_vectorizer.fit_transform(documents)

print("Bag of Words (Count Occurrence):")
print("Vocabulary:", count_vectorizer.get_feature_names_out())
print(bow_counts.toarray())

print("\n" + "="*60)

bow_array = bow_counts.toarray().astype(float)
normalized_bow = bow_array / bow_array.sum(axis=1, keepdims=True)

print("Bag of Words (Normalized Count Occurrence):")
print(normalized_bow)

print("\n" + "="*60)

tfidf_vectorizer = TfidfVectorizer(lowercase=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print("TF-IDF:")
print("Vocabulary:", tfidf_vectorizer.get_feature_names_out())
print(tfidf_matrix.toarray())

print("\n" + "="*60)

tokenized_docs = [doc.lower().split() for doc in documents]

word2vec_model = Word2Vec(
    sentences=tokenized_docs,
    vector_size=50,
    window=3,
    min_count=1,
    workers=4
)

print("Word2Vec Embedding Example:")
print("Vector for word 'sports':")
print(word2vec_model.wv['sports'])
