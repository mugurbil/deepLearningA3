#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=12:00:00
#PBS -l mem=64GB
#PBS -N M_AND_M

read N

for((i=0;i<$N;i++))
    do
        read line
        th classify.lua -input "$line"
    done
