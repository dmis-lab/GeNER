#!/bin/bash

FILEPATH=$1 ;
AUTOPHARSE_DIR=./autophrase
CURRENT_PATH="$(realpath ./)"

cd ${AUTOPHARSE_DIR}

for FILE in ${CURRENT_PATH}/$FILEPATH/*.json; do
	bash phrasal_segmentation.sh "${FILE: 0: -5}".raw &&
	mv models/DBLP/segmentation.txt "${FILE: 0: -5}".autophrase
done





