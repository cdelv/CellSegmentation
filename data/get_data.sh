#!/bin/bash
set -e

rm -rf train test
mkdir -p train test

train_url=(
http://data.celltrackingchallenge.net/training-datasets/BF-C2DL-HSC.zip
http://data.celltrackingchallenge.net/training-datasets/BF-C2DL-MuSC.zip
http://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip
http://data.celltrackingchallenge.net/training-datasets/Fluo-C2DL-Huh7.zip
http://data.celltrackingchallenge.net/training-datasets/Fluo-C2DL-MSC.zip
http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip
http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DL-HeLa.zip
http://data.celltrackingchallenge.net/training-datasets/PhC-C2DH-U373.zip
http://data.celltrackingchallenge.net/training-datasets/PhC-C2DL-PSC.zip
http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-SIM+.zip
)

test_url=(
http://data.celltrackingchallenge.net/test-datasets/BF-C2DL-HSC.zip
http://data.celltrackingchallenge.net/test-datasets/BF-C2DL-MuSC.zip
http://data.celltrackingchallenge.net/test-datasets/DIC-C2DH-HeLa.zip
http://data.celltrackingchallenge.net/test-datasets/Fluo-C2DL-Huh7.zip
http://data.celltrackingchallenge.net/test-datasets/Fluo-C2DL-MSC.zip
http://data.celltrackingchallenge.net/test-datasets/Fluo-N2DH-GOWT1.zip
http://data.celltrackingchallenge.net/test-datasets/Fluo-N2DL-HeLa.zip
http://data.celltrackingchallenge.net/test-datasets/PhC-C2DH-U373.zip
http://data.celltrackingchallenge.net/test-datasets/PhC-C2DL-PSC.zip
http://data.celltrackingchallenge.net/test-datasets/Fluo-N2DH-SIM+.zip
)

export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE

for url in ${train_url[@]}; do
    wget $url -O $(basename $url)
    unzip $(basename $url) -d train
    mv $(basename $url) train
done

for url in ${test_url[@]}; do
    wget $url -O $(basename $url)  
    unzip $(basename $url) -d test
    mv $(basename $url) test
done