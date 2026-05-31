#!/bin/bash
# Saves files to gdrive every 3 minutes.

while true
do 
    cp "$@" "/content/drive/MyDrive/"
    echo "pushed"
    sleep 180
done