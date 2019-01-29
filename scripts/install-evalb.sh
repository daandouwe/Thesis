#!/usr/bin/env bash

echo 'Downloading EVALB...'
wget https://nlp.cs.nyu.edu/evalb/EVALB.tgz
tar -xzf EVALB.tgz
rm EVALB.tgz

cd EVALB
sed -i -e 's/#include <malloc.h>/\/* #include <malloc.h> *\//g' evalb.c  # remove <malloc.h> include
make
cd ..

echo 'Succesfully installed EVALB.'
