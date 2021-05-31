#!/bin/bash
# script to download and set up data

function downloadOpenSubtitlesCorpus() {
  if [ -d "./raw/$1" ] 
  then
    echo "Directory for $1 already exists, skipping download."
  else
    # download relevant OpenSubtitles corpus files
    curl -L https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/$1.txt.zip -o ./raw/$1.txt.zip

    # unzip downloaded file
    unzip ./raw/$1.txt.zip -d ./raw/$1

    # remove zipped files
    rm ./raw/$1.txt.zip
  fi
}

# download data
downloadOpenSubtitlesCorpus "en-ja"
downloadOpenSubtitlesCorpus "en-fr"
downloadOpenSubtitlesCorpus "en-es"

# assemble a four way corpus with identical documents for each language
python construct_four_way_parallel_corpus.py