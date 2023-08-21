conda create -n bahn_gpu
conda activate bahn_gpu
conda install -c conda-forge spacy
conda install -c conda-forge cupy
conda install -c conda-forge spacy-transformers
# packages only available via pip
pip install spacy-lookups-data
python -m spacy download de_core_news_sm


python -m spacy train config.cfg --paths.train ./own_bahn_sentiment_train.spacy --paths.dev ./own_bahn_sentiment_dev.spacy
