#!/bin/bash

export WINDOW=5
export OUT_DIR=$1/processed
export MDL_DIR=$1/models

export LUA_PATH="$LUA_PATH;$ABS/?.lua"

#bash $ABS/prep_torch_data.sh $2

mkdir -p $MDL_DIR

th -i test.lua -titleDir  $OUT_DIR/train/title/ \
 -articleDir  $OUT_DIR/train/article/ \
 -validArticleDir  $OUT_DIR/valid.filter/article/ \
 -validTitleDir  $OUT_DIR/valid.filter/title/ \
