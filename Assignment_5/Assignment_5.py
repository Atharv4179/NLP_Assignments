# Import required libraries
import nltk
from nltk.tokenize import word_tokenize, TweetTokenizer, MWETokenizer
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Sample text (unique + realistic)
text = """
India won the final match!!!
Players were running, celebrating, and tweeting #Victory2026.
The Prime Minister said New Delhi will host the event.
Artificial Intelligence and Machine Learning are transforming sports analytics.
"""

print("Original Text:\n", text)

# -------------------------------------------------
# 1. TOKENIZATION
# -------------------------------------------------

# a) Whitespace Tokenization
whitespace_tokens = text.split()
print("\nWhitespace Tokenization:\n", whitespace_tokens)

# b) Punctuation-based Tokenization
punctuation_tokens = re.findall(r'\b\w+\b', text)
print("\nPunctuation-based Tokenization:\n", punctuation_tokens)

# c) Treebank Tokenization
treebank_tokens = word_tokenize(text)
print("\nTreebank Tokenization:\n", treebank_tokens)

# d) Tweet Tokenization
tweet_tokenizer = TweetTokenizer()
tweet_tokens = tweet_tokenizer.tokenize(text)
print("\nTweet Tokenization:\n", tweet_tokens)

# e) MWE Tokenization
mwe_tokenizer = MWETokenizer([('Artificial', 'Intelligence'),
                              ('Machine', 'Learning'),
                              ('New', 'Delhi')], separator='_')
mwe_tokens = mwe_tokenizer.tokenize(word_tokenize(text))
print("\nMWE Tokenization:\n", mwe_tokens)

# -------------------------------------------------
# 2. STEMMING
# -------------------------------------------------

porter = PorterStemmer()
snowball = SnowballStemmer("english")

sample_words = ["running", "celebrating", "transforming", "analytics"]

porter_stems = [porter.stem(word) for word in sample_words]
snowball_stems = [snowball.stem(word) for word in sample_words]

print("\nSample Words:", sample_words)
print("Porter Stemmer Output:", porter_stems)
print("Snowball Stemmer Output:", snowball_stems)

# -------------------------------------------------
# 3. LEMMATIZATION
# -------------------------------------------------

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word, pos='v') for word in sample_words]

print("\nLemmatization Output:", lemmatized_words)
