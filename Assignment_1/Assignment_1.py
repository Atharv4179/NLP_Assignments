import nltk


nltk.download('punkt')
nltk.download('wordnet')

from nltk.tokenize import (
    WhitespaceTokenizer,
    wordpunct_tokenize,
    TreebankWordTokenizer,
    TweetTokenizer,
    MWETokenizer
)
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer


text = "I went on a vacation to my hometown and enjoyed delicious food with my family"

print("Original Text:")
print(text)
print("-" * 50)


whitespace_tokenizer = WhitespaceTokenizer()
print("Whitespace Tokenization:")
print(whitespace_tokenizer.tokenize(text))
print()


print("Punctuation-based Tokenization:")
print(wordpunct_tokenize(text))
print()


treebank_tokenizer = TreebankWordTokenizer()
treebank_tokens = treebank_tokenizer.tokenize(text)
print("Treebank Tokenization:")
print(treebank_tokens)
print()


tweet_tokenizer = TweetTokenizer()
print("Tweet Tokenization:")
print(tweet_tokenizer.tokenize(text))
print()


mwe_tokenizer = MWETokenizer([('delicious', 'food'), ('my', 'family')], separator='_')
print("MWE Tokenization:")
print(mwe_tokenizer.tokenize(text.split()))
print("-" * 50)


porter = PorterStemmer()
snowball = SnowballStemmer("english")

print("Porter Stemmer:")
print([porter.stem(word) for word in treebank_tokens])
print()

print("Snowball Stemmer:")
print([snowball.stem(word) for word in treebank_tokens])
print("-" * 50)


lemmatizer = WordNetLemmatizer()
print("Lemmatization:")
print([lemmatizer.lemmatize(word) for word in treebank_tokens])
