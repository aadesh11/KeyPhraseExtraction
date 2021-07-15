from sklearn.metrics.pairwise import cosine_similarity
from keyphrase.normalize import *


class RunAlgo():
    def __init__(self, top_k=5, text_embedding=None, candidate_embeddings=None, candidates=None):
        self.text_embedding = text_embedding
        self.candidate_embeddings = candidate_embeddings
        self.candidate_words = candidates
        self.top_k = top_k

    #consine similarity between candidates embeddings and text embeddings, find top keywords

    #Just select top k candidates keywords
    def select_topn_candidates(self,):
        distances = cosine_similarity(self.text_embedding, self.candidate_embeddings)
        keywords = [self.candidate_words[index] for index in distances.argsort()[0][-self.top_k:]]
        return keywords


    # Use MMR(Maximal margin relevance) score to get top keywords, it will have better diversity and relevance
    def mmr(self, beta=0.55, alias_threshold=0.8):
        """Implementation of Maximal Marginal Relevance to get top N relevant keywords to text
        Args:
            self.text_embedding: embedding of original text (from where keywords are extracted)
            self.candidate_words: keywords (noun phrases) selected from text from where we have to choose broad and relevant keywords
            candidate_embeddings: embeddings of given keywords
            beta: hyper-parameter for MMR score calculations (controls tradeoff between informativeness and diversity)
            top_k: number of top keywords to extract (will return less keywords if len(keywords) < top_k)
            alias_threshold(very similar): threshold for cosine similarities (controls alias keyword pairs assignment)
        Returns:
            top_keywords: selected top keywords
            relevance: relevance values for these keywords (relevance of keyword to original text)
            aliases_keywords: aliases for each keyword
        """
        # calculate similarities of keywords with text and between keywords
        text_sims = cosine_similarity(self.candidate_embeddings, self.text_embedding)
        keyword_sims = cosine_similarity(self.candidate_embeddings)

        # normalize cosine similarities
        text_sims_norm = standard_normalize_cosine_similarities(text_sims)
        keyword_sims_norm = max_normalize_cosine_similarities_pairwise(keyword_sims)

        # keep indices of selected and unselected keywords in list
        selected_keyword_indices = []
        unselected_keyword_indices = list(range(len(self.candidate_words)))

        # find the most similar keyword (using original cosine similarities)
        best_idx = np.argmax(text_sims)
        selected_keyword_indices.append(best_idx)
        unselected_keyword_indices.remove(best_idx)

        # do top_n - 1 cycle to select top N keywords
        for _ in range(min(len(self.candidate_words), self.top_k) - 1):
            unselected_keyword_distances_to_text = text_sims_norm[unselected_keyword_indices, :]
            unselected_keyword_distances_pairwise = keyword_sims_norm[unselected_keyword_indices][:,
                                                    selected_keyword_indices]

            # if dimension of keywords distances is 1 we add additional axis to the end
            if unselected_keyword_distances_pairwise.ndim == 1:
                unselected_keyword_distances_pairwise = np.expand_dims(unselected_keyword_distances_pairwise, axis=1)

            # find new candidate with
            idx = int(np.argmax(
                beta * unselected_keyword_distances_to_text - (1 - beta) * np.max(unselected_keyword_distances_pairwise,
                                                                                  axis=1).reshape(-1, 1)))
            best_idx = unselected_keyword_indices[idx]

            # select new best keyword and update selected/unselected keyword indices list
            selected_keyword_indices.append(best_idx)
            unselected_keyword_indices.remove(best_idx)

        # calculate relevance using original (not normalized) cosine similarities of keywords to text
        relevance = max_normalize_cosine_similarities(text_sims[selected_keyword_indices]).tolist()
        aliases_keywords = get_alias_keywords(keyword_sims[selected_keyword_indices, :], self.candidate_words, alias_threshold)

        top_keywords = [self.candidate_words[idx] for idx in selected_keyword_indices]

        # for showing vectors in space
        embs = self.candidate_embeddings + self.text_embedding

        plot(embs, self.candidate_words)

        return top_keywords, relevance, aliases_keywords