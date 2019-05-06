#!/usr/bin/env bash

FILE=02-21.10way.clean

cat $FILE | cut -c6- | rev | cut -c2- | rev > $FILE.notop
