#!/usr/bin/env

# Dowloads EVALB on Lisa.

cd  # Install EVALB in home directory.
echo 'Downloading EVALB.'
wget https://nlp.cs.nyu.edu/evalb/EVALB.tgz
tar -xzf EVALB.tgz
rm EVALB.tgz
cd EVALB
# Remove <malloc.h> include.
sed -i -e 's/#include <malloc.h>/\/* #include <malloc.h> *\//g' evalb.c
make
