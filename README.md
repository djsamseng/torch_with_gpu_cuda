# CTorch

## Run C++ Torch with CUDA GPU

```bash
mkdir build
cd build
# If needed: create conda environment for torch dependencies
conda create --name torch python=3.8
# Activate conda environment
conda activate torch
# If needed: Install necessary libraries
conda install -c anaconda cudatoolkit
conda install -c conda-forge cudatoolkit-dev
conda install -c anaconda cudnn
# If needed: Download libtorch with correct CUDA from pytorch.org and unzip. The path to which is used below (/home/samuel/dev/ctorch/libtorch)
# If needed: Create a symlink to runtime data (data will have MNIST/raw/train-images-idx3-ubyte MNIST/raw/train-labels-idx1-ubyte etc.)
ln -s /path/to/data/ .
# Create build directory
mkdir build
cd build
# Generate build files
cmake -DCMAKE_PREFIX_PATH=/home/samuel/dev/ctorch/libtorch ..
# Build
cmake --build . --config Release
# Run
./ctorch-app
```
