import re

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

from model_wrappers.preprocessing import Preprocessor

class Isaac(Preprocessor):
    _snowball_stemmer = SnowballStemmer(language='english')
    _stop_words = stopwords.words('english')

    def __init__(self) -> None:
        super().__init__()

    def _preprocess(self, raw_text: str) -> str:
        text = self._clean_data(raw_text)
        text = self._remove_stop_words(text)
        text = self._sbstemming(text)
        text = ' '.join(text)
        text = text.encode('utf-8')
        return text

    def _clean_data(self, text):
        text = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", "", text) ## removes all punctuation
        text = re.sub("[^a-zA-Z0-9!?%()<>{}:;\"\',.\[\] ]", "", text)
        text = text.lower()
        text = word_tokenize(text)
        return text

    def _remove_stop_words(self, text):
        return [word for word in text if word not in Isaac._stop_words]

    def _sbstemming(self, text):
        try:
            text = [Isaac._snowball_stemmer.stem(word) for word in text]
            text = [word for word in text if len(word) > 1] # make sure we have no 1 letter words
        except IndexError:
            pass
        return text
