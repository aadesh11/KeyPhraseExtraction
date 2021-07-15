import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


def standard_normalize_cosine_similarities(cosine_similarities):
    """Normalized cosine similarities"""
    # normalize into 0-1 range
    cosine_sims_norm = (cosine_similarities - np.min(cosine_similarities)) / (
        np.max(cosine_similarities) - np.min(cosine_similarities)
    )

    # standardize and shift by 0.5
    cosine_sims_norm = 0.5 + (cosine_sims_norm - np.mean(cosine_sims_norm)) / np.std(
        cosine_sims_norm
    )

    return cosine_sims_norm


def max_normalize_cosine_similarities_pairwise(cosine_similarities):
    """Normalized cosine similarities of pairs which is 2d matrix of pairwise cosine similarities"""
    cosine_sims_norm = np.copy(cosine_similarities)
    np.fill_diagonal(cosine_sims_norm, np.NaN)

    # normalize into 0-1 range
    cosine_sims_norm = (
        cosine_similarities - np.nanmin(cosine_similarities, axis=0)
    ) / (
        np.nanmax(cosine_similarities, axis=0) - np.nanmin(cosine_similarities, axis=0)
    )

    # standardize shift by 0.5
    cosine_sims_norm = 0.5 + (
        cosine_sims_norm - np.nanmean(cosine_sims_norm, axis=0)
    ) / np.nanstd(cosine_sims_norm, axis=0)

    return cosine_sims_norm


def max_normalize_cosine_similarities(cosine_similarities):
    """Normalize cosine similarities using max normalization approach"""
    return 1 / np.max(cosine_similarities) * cosine_similarities.squeeze(axis=1)


def get_alias_keywords(keyword_sims, keywords, threshold):
    """Find keywords in selected list that are aliases (very similar) to each other"""
    similarities = np.nan_to_num(keyword_sims, 0)
    sorted_similarities = np.flip(np.argsort(similarities), 1)

    aliases = []
    for idx, item in enumerate(sorted_similarities):
        alias_for_item = []
        for i in item:
            if similarities[idx, i] >= threshold:
                alias_for_item.append(keywords[i])
            else:
                break
        aliases.append(alias_for_item)

    return aliases


def plot(embs, texts):
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(embs)
    data = pd.DataFrame(
        {
            "v1": vectors_2d[:, 0],
            "v2": vectors_2d[:, 1],
        }
    )
    ax = sns.scatterplot(x=data.v1, y=data.v2)  # style=data.type, hue=data.type

    for i, text in enumerate(zip(texts)):
        if len(text) > 20:
            text = text[:20] + "..."
        ax.annotate(text, (vectors_2d[i, 0], vectors_2d[i, 1]))

    plt.show()
