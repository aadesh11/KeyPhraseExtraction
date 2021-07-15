from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from keyphrase.preprocess import clean_token_text
import spacy


class ExtractCandidates():
    def __init__(self, ngram_range, texts):
        nltk.download('stopwords')
        nlp = spacy.load('en_core_web_sm')
        #load stopwords
        self.stop_words = set(list(stopwords.words("english")))
        self.texts = clean_token_text(texts)
        self.doc = nlp(self.texts)
        self.ngram_range = ngram_range #(2, 3)
        self.is_stop_word = lambda token: token.is_stop or token.lower_ in self.stop_words or token.lemma_ in self.stop_words

    def gen_ngrams(self):
        all_ngrams = []
        tokens = [token.text for token in self.doc if not self.is_stop_word(token)]
        for ngram in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(tokens) - ngram + 1):
                n_gram_phrase = " ".join(tokens[i:i + ngram])

                all_ngrams.append(n_gram_phrase)
        return all_ngrams

    # Extract candidate words/phrases using n_grams
    def extract_all_candidates_from_text_using_count_vector(self):
        count = CountVectorizer(ngram_range=self.ngram_range, stop_words=self.stop_words).fit([self.texts])
        all_candidates = count.get_feature_names()
        return all_candidates

    #Extract only noun phrase chunks
    def extract_noun_phrases(self,):
        np_candidates = set()
        for noun_chunk in self.doc.noun_chunks:
            chunk_text = noun_chunk.text.strip().replace("\n", "").lower()
            chunk_text_tokens = chunk_text.split()
            chunk_text_tokens = [clean_token_text(token_text) for token_text in chunk_text_tokens
                                 if token_text not in self.stop_words]
            if len(chunk_text_tokens) > 0:
                np_candidates.add(' '.join(chunk_text_tokens))
        return np_candidates

    # Extract only noun tokens
    def extract_single_nouns(self,):
        nouns = set()
        for token in self.doc:
            is_stop = self.is_stop_word(token)
            if token.pos_ == "NOUN" and len(token.text) > 1 and not is_stop:
                nouns.add(token.text)
        return nouns

    # extract noun and noun phrase from the text
    def create_candidate_with_noun(self,):
        nouns = self.extract_single_nouns()
        noun_phrases = self.extract_noun_phrases()
        #get union of nouns and noun-phrases
        all_nouns = nouns.union(noun_phrases)
        return all_nouns

    # entry point to get all the candidates key-phrase
    def get_candidates(self, only_nouns=False, using_count_vector=True):
        if using_count_vector:
            # extract all n_grams from the text using CountVectorizer (doesn't split sentence properly)
            n_gram_candidates = self.extract_all_candidates_from_text_using_count_vector()
        else:
            n_gram_candidates = self.gen_ngrams()

        if not only_nouns:
            return n_gram_candidates

        # extract all noun and noun phrase from the text
        all_nouns_and_np = self.create_candidate_with_noun()

        # filter only those n_grams which contains noun or noun-phrase
        filtered_candidates = list(filter(lambda candidate: candidate in all_nouns_and_np, n_gram_candidates))
        return filtered_candidates