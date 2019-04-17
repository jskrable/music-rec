# usr/bin/sh
# Bash script to setup environment on remote server

git clone https://github.com/jskrable/music-rec
cd music-rec

# mkdir data && cd data
# wget http://static.echonest.com/millionsongsubset_full.tar.gz
# tar -xvzf millionsongsubset_full.tar.gz

sudo mkdir /mnt/snap
sudo mount -t ext4 /dev/xvdf /mnt/snap

mkdir model
mkdir logs


