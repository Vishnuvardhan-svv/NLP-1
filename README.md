CareerPilot AI analyzes cv and job descriptions to identify technical skill gaps.
It uses a local skill taxonomy for extraction and matching, plus TF-IDF, Word2Vec, and Transformer similarity.
The pretrained Sentence Transformer runs locally and produces embeddings without hosted AI services.
It calculates exact and weighted skill coverage and suggests a prerequisite-ordered learning roadmap.
A simple Streamlit interface presents the analysis, comparisons, and recommendations.


Deployed Website Link - https://hxfllnkyveemdhuo6dfb8g.streamlit.app/

## Streamlit Community Cloud deployment

Select Python **3.11** in Community Cloud's Advanced settings when deploying. The supplied build log shows Python 3.14.10, where Gensim's native extension fails to compile. Community Cloud requires deleting and redeploying the existing app to change its Python version. The CPU-only PyTorch wheel is configured in `requirements.txt` to avoid downloading CUDA packages on Cloud.
