import re
from nltk.tokenize import wordpunct_tokenize

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u00a0", " ").lower()
    text = re.sub(r"[^\w+#./-]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text: str) -> list[str]:
    return [token for token in wordpunct_tokenize(clean_text(text)) if token.strip()]
