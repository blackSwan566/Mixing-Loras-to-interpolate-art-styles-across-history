#! /bin/bash

#Define variables
SOURCE_FILE= "/Users/fionalau/Desktop/Mixing Loras for interpolating art styles across history/Mixing-Loras-to-interpolate-art-styles-across-history/renaissance_dataset.tar"
REMOTE_USER= "lauf"
REMOTE_HOST= "remote.cip.ifi.lmu.de"
REMOTE_PATH="/home/l/lauf/work/Mixing-Loras-to-interpolate-art-styles-across-history"
REMOTE_PATH ="/remote/path/"

resync -avh $SOURCE_FILE $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH

if [$? -eq 0]; then
  echo "Successfully transferred"
else
  echo "Error during rsync"
fi