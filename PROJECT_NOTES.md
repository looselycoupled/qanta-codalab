

# Development

## Interactive Bash in Container

You can startup the container with a bash session using the following command.  Once inside your bash session you can start the webserver using the `run.sh` script.

    docker-compose run -it --service-ports  entilzha/quizbowl /bin/bash

As a reminder, you can test a request from outside the container (on your host computer) using HTTPie.

    http POST http://0.0.0.0:4861/api/1.0/quizbowl/act text='Name the the inventor of general relativity and the photoelectric effect'


# Setup

## Python Dependencies

If you are not using Docker, you can get the code working locally by installing the dependencies using `pip`.

    pip install pytorch numpy scipy pandas scikit-learn nltk requests click flask

## Pretrained Embeddings

To download the pretrained word embeddings from Google use the following command and place in the data directory.

    wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
    gunzip GoogleNews-vectors-negative300.bin.gz
