from transformers import AutoModel, AutoTokenizer


class TransformerModels:
    def __init__(self, model_name="xlm-roberta-base"):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Build model and get embeddings from it.
    def build_model(self, texts=None):
        # texts: can be candidates phrase sentences or a given sentences for which we need to extract key-phrase
        texts_tokens = self.tokenizer(texts, padding=True, return_tensors="pt")
        texts_embeddings = self.model(**texts_tokens)["pooler_output"]
        return texts_embeddings.detach().numpy()
