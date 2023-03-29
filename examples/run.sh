#!/bin/bash

CONFIG=SN2004C_config.ini
OUTPUT_DIR=./

snart \
  --config-file ${CONFIG} \
  --output-dir ${OUTPUT_DIR} \
  --make-plots
