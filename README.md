# installation


$ sudo apt install cmake build-essential

# GMP (https://ftp.gnu.org/gnu/gmp/)
# under third_party/
# make sure not in conda environment

```bash
$ sudo apt install m4
$ cd gmp-6.2.1
$ ./configure SHARED=on 
$ make
$ make check  # GMP developers strongly recommend doing this
$ (sudo) make install
$ sudo ldconfig 
```

# NTL (https://libntl.org/download.html)

```bash
cd ntl-11.5.1/src
./configure SHARED=on NTL_GMP_LIP=on  ## conda등 가상 환경 변수가 잡혀있을 경우 gmp를 못 찾을 수 있으니 주의
make
make check  # optional
(sudo) make install
(sudo) ldconfig
```

### 1.3 Anaconda env

`conda install ipython numpy jupyter pybind11 scipy scikit-learn tqdm matplotlib`  
`pip install torch torchvision torchaudio`   
`pip install opencv-python pyqt5`  


### Install the package 
run `$ ./build.sh`
