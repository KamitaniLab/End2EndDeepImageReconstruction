#!/bin/bash
#
# Download CaffeNet
#
# Usage:
#
#   ./downloadnet.sh caffenet
#

## Functions

function download_file () {
    dlurl=$1
    dlpath=$2
    dldir=$(dirname $dlpath)
    dlfile=$(basename $dlpath)

    [ -d $didir ] || mkdir $dldir
    if [ -f $dldir/$dlfile ]; then
        echo "$dlfile has already been downloaded."
    else
        curl -o $dldir/$dlfile $dlurl
    fi
}

## Main

target=$1

if [ "$target" = '' ]; then
    echo "Please specify the network ('caffenet') to be downloaded."
    exit 1
fi

# CaffeNet
if [ "$target" = 'caffenet' ]; then
    output=./bvlc_reference_caffenet.caffemodel
    srcurl=http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel

    [ -f $output ] && echo "$output already exists." && exit 0
    
    download_file $srcurl $output

    echo "$output saved."
    exit 0
fi

# Unknown target
echo "Unknown network $target"
exit 1
