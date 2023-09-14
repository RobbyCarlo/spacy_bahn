conda create -n bahn_gpu
conda activate bahn_gpu
conda install -c conda-forge spacy
conda install -c conda-forge cupy
conda install -c conda-forge spacy-transformers
# packages only available via pip
pip install spacy-lookups-data
python -m spacy download de_core_news_sm

#### Paramter für das Programm (kann man auf Spacy nachschauen) 
--input-path own_bahn_sentiment_test.jsonl --label positive --label negative --language de

python -m spacy train config.cfg --paths.train ./own_bahn_sentiment_train.spacy --paths.dev ./own_bahn_sentiment_dev.spacy --gpu-id 0 -o model

python -m spacy benchmark accuracy ./model/model-best own_bahn_sentiment_test.spacy --gpu-id 0 -o metrics.json



