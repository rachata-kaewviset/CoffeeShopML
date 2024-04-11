import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

data = pd.read_csv('fakenews.csv')

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'  # Adjective
    elif tag.startswith('V'):
        return 'v'  # Verb
    elif tag.startswith('N'):
        return 'n'  # Noun
    elif tag.startswith('R'):
        return 'r'  # Adverb
    else:
        return 'n'

lemmatizer = WordNetLemmatizer()

data['text'] = data['text'].str.lower()
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 1]))
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if not re.match(r'.*\d.*', word)]))
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.isalnum()]))
stop_words = set(stopwords.words('english'))
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
data['text'] = data['text'].apply(lambda x: pos_tag(word_tokenize(x)))

data['text'] = data['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in x]))

cleaned_file_path = "cleaned_fakenews.csv"
data.to_csv(cleaned_file_path, index=False)
