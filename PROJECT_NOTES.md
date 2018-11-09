# Setup


## Python Dependencies

If you are not using Docker, you can get the code working locally by installing the dependencies using `pip`.

    pip install pytorch numpy scipy pandas scikit-learn nltk requests click flask

## Pretrained Embeddings

To download the pretrained word embeddings from Google use the following command and place in the data directory.

    wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
    gunzip GoogleNews-vectors-negative300.bin.gz
