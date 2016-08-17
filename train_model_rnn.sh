#!/bin/bash

export WINDOW=5
export LDC=/run/media/julien/MyPassport/LDC/
export OUT_DIR=$LDC/processed
export MDL_DIR=$LDC/models

export LUA_PATH="$LUA_PATH;$ABS/?.lua"

#bash $ABS/prep_torch_data.sh $2

mkdir -p $MDL_DIR

th -i $ABS/summary/train_rnn.lua -titleDir  $OUT_DIR/train/title/ \
 -articleDir  $OUT_DIR/train/article/ \
 -modelFilename  $MDL_DIR/$2 \
 -miniBatchSize  64 \
 -embeddingDim  64 \
 -hiddenSize  64 \
 -epochs  20 \
 -learningRate 0.1 \
 -validArticleDir  $OUT_DIR/valid.filter/article/ \
 -validTitleDir  $OUT_DIR/valid.filter/title/ \
 -window  $WINDOW \
 -printEvery   100 \
