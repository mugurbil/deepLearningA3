#!/bin/bash

qsub -I -l nodes=1:ppn=1:gpus=1:titan -l mem=8GB

git clone https://github.com/markisus/Deep-Learning-Assignment-3.git part1

cd ~
git clone https://github.com/mugurbil/deepLearningA3.git M_AND_M
mv part1/A3_baseline.lua M_AND_M/A3_baseline.lua
mv part1/A3_skeleton.lua M_AND_M/A3_skeleton.lua

cd M_AND_M/

th M_AND_M_model.lua
./M_AND_M_model.sh

