import re

"""
To add any pre-process steps here.
"""


# To clean the text sentences
def clean_token_text(token_text):
    tokens = token_text.split()
    p = r"[^A-Za-z0-9]\Z"
    tokens = [re.sub(p, "", token) for token in tokens]
    tokens = [token for token in tokens if len(token) > 1]
    return " ".join(tokens)
