#!/bin/bash
#
# Download the preprocessed fmri data
#
# Usage:
#
#   ./download_fmri.sh
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

download_file https://ndownloader.figshare.com/files/14830643 ./sub-01_perceptionNaturalImageTraining_original_VC.h5

download_file https://ndownloader.figshare.com/files/14830631 ./sub-01_perceptionNaturalImageTest_original_VC.h5

download_file https://ndownloader.figshare.com/files/14830712 ./sub-02_perceptionNaturalImageTraining_original_VC.h5

download_file https://ndownloader.figshare.com/files/14830697 ./sub-02_perceptionNaturalImageTest_original_VC.h5

download_file https://ndownloader.figshare.com/files/14830862 ./sub-03_perceptionNaturalImageTraining_original_VC.h5

download_file https://ndownloader.figshare.com/files/14830856 ./sub-03_perceptionNaturalImageTest_original_VC.h5


echo "done!"


