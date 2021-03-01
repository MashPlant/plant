#!/bin/bash

cd isl
if [ ! -d "build" ]; then
  mkdir build
fi
touch aclocal.m4 Makefile.am Makefile.in
./configure --prefix=$PWD/build/ --with-int=gmp
make -j 16
make install
