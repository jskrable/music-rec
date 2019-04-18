# usr/bin/sh
# Bash script to setup environment on remote server

# git clone https://github.com/jskrable/music-rec
# cd music-rec

mkdir data && cd data
wget http://static.echonest.com/millionsongsubset_full.tar.gz
tar -xvzf millionsongsubset_full.tar.gz

cd ..
mkdir model
mkdir logs

# May need sudo here
chmod +x lib/main.py
chmod +x api/api.py

