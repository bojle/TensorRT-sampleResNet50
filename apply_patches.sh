#!/bin/bash

patch -p 1 -i make.patch ../Makefile
patch -p 1 -i make.config.patch ../Makefile.config
