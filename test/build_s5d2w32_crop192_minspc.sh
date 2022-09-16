#!/usr/bin/bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t autopet_s5d2w32_crop192_minspc "$SCRIPTPATH" -f $SCRIPTPATH/Dockerfile_s5d2w32_crop192_minspc --no-cache
