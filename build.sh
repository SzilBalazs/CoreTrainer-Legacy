#!/bin/bash

g++ -Wall -Wextra -shared -fPIC -O3 -march=native -o dataloader.so dataloader.cpp
g++ -Wall -Wextra -O3 -march=native -o dataparser dataparser.cpp
g++ -Wall -Wextra -O3 -o netconverter netconverter.cpp
