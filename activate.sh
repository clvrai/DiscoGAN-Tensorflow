#!/bin/bash

ROOT=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

export PATH=$ROOT/TensorArtist/bin:$PATH
export TART_DIR_DATA=$ROOT/data/
export TART_DIR_DUMP=$ROOT/dump/
