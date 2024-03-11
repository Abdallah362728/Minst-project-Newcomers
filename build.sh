#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# Define the source directory and Eigen library directory
SRC_DIR=./src
EIGEN_DIR=./eigen-3.4.0

# Clean up previous build artifacts
echo "Cleaning up previous build artifacts..."
rm -f $SRC_DIR/*.o $SRC_DIR/a.out $SRC_DIR/my_executable

# Compilation command with g++
echo "Building project..."
g++ -I $EIGEN_DIR -std=c++11 $SRC_DIR/*.cpp -o $SRC_DIR/my_executable

echo "Build completed successfully."
