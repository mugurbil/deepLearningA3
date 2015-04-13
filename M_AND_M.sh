#!/bin/bash

qsub -I -l nodes=1:ppn=1:gpus=1:titan -l mem=8GB

git clone https://github.com/markisus/Deep-Learning-Assignment-3.git part1

cd ~
mkdir M_AND_M
mv part1/A3_baseline.lua M_AND_M/A3_baseline.lua
mv part1/A3_skeleton.lua M_AND_M/A3_skeleton.lua

cd M_AND_M/

wget http://cims.nyu.edu/~mu388/M_AND_M.pdf

git clone https://github.com/mugurbil/deepLearningA3.git

th M_AND_M_model.lua
./M_AND_M_model.sh

