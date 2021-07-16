[![Build Status](https://travis-ci.com/LPMP/BDD.svg?branch=main)](https://travis-ci.com/LPMP/BDD)

# BDD
An integer linear program solver using a Lagrange decomposition into binary decision diagrams. Lagrange multipliers are updated through dual block coordinate ascent.

## Installation

`git clone https://github.com/LPMP/BDD`

`git submodule update --remote --recursive --init`

Then continue with creating a build folder and use cmake:

`mkdir build && cd build && cmake ..`
