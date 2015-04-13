#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=12:00:00
#PBS -l mem=64GB
#PBS -N M_AND_M

cd ~

git clone https://github.com/markisus/Deep-Learning-Assignment-3.git part1

git clone https://github.com/mugurbil/deepLearningA3.git M_AND_M
mv part1/A3_baseline.lua M_AND_M/A3_baseline.lua
mv part1/A3_skeleton.lua M_AND_M/A3_skeleton.lua

cd M_AND_M/

wget https://cims.nyu.edu/~mu388/model.net

