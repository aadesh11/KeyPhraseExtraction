from keyphrase.extract_candidates import ExtractCandidates
from keyphrase.mmr_algo import RunAlgo
from keyphrase.transformer_models import TransformerModels

"""
Follow the idea of https://arxiv.org/pdf/1801.04470v3.pdf, in addition to user can choose to select candidates key-phrase
as ngram + noun/noun-phrase or only n_gram based key-phrase selection.
"""


def main():
    text = """
                 In natural language processing, semantic role labeling is the process that assigns labels to words or
                 phrases in a sentence that indicates their semantic role in the sentence, such as that of an agent, goal,
                 or result. It serves to find the meaning of the sentence. To do this, it detects the arguments associated
                 with the predicate or verb of a sentence and how they are classified into their specific roles.
                 A common example is the sentence "Mary sold the book to John". The agent is Mary, the predicate is
                 "sold" the theme is "the book" and the recipient is "John". Another example is how "the
                 book belongs to me" would need two labels such as "possessed" and "possessor" and "the book was sold to John" would
                 need two other labels such as theme and recipient, despite these two clauses being similar to "subject" and "object"
                 functions.
              """
    NGRAM_RANGE = (2, 3)
    TOP_K = 5
    # hyper-parameter for MMR score calculations (controls tradeoff between informativeness and diversity)
    BETA = 0.55
    ALIAS_THRESHOLD = 0.8
    # Filter only noun or noun-phrase candidate keyphrase
    only_noun_and_np = True

    # get all the key-phrase from the text
    extract = ExtractCandidates(ngram_range=NGRAM_RANGE, texts=text)
    candidate_keyphrase = extract.get_candidates(only_nouns=only_noun_and_np)

    # get embeddings from model for key-phrase and text
    model = TransformerModels(model_name="xlm-roberta-base")
    candidates_embeddings = model.build_model(texts=candidate_keyphrase)
    text_embeddings = model.build_model(texts=[text])

    # get keywords using normal or MMR based key-phrase
    run_algo = RunAlgo(
        top_k=TOP_K,
        text_embedding=text_embeddings,
        candidate_embeddings=candidates_embeddings,
        candidates=candidate_keyphrase,
    )
    keywords_without_mmr = run_algo.select_topn_candidates()
    keywords_wth_mmr, relavance, aliases_keywords = run_algo.mmr(
        beta=BETA, alias_threshold=ALIAS_THRESHOLD
    )

    print("Keyword without MMR: ", keywords_without_mmr)
    print("Keyword with MMR: ", keywords_wth_mmr)
    print("Relavance Score: ", relavance)
    print("Aliases Keywords ", aliases_keywords)


if __name__ == "__main__":
    main()
