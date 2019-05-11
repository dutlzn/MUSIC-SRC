#!/bin/sh
rsync -z -P -l -r --exclude "checkpoints" --exclude "runs" --exclude ".git" --exclude "messages.log" youchen@10.7.15.44:/home/youchen/MusicGenreClassification/ .
