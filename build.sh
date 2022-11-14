build_fpga=false
build_cuda=false


export ROOT_DIR=$PWD
#export CGBN_INCLUDE_DIR=$ROOT_DIR/third_party/CGBN/include

cd $ROOT_DIR/HEAAN/lib/
make clean && make  -j 4

if [ "$build_fpga" = true ] ; then
    cd $ROOT_DIR/HEAAN_fpga/lib/
    make clean && make -j 4
fi

if [ "$build_cuda" = true ] ; then
    cd $ROOT_DIR/HEAAN_cuda/lib/
    make clean && make -j 4
fi


cd $ROOT_DIR
pip install -e .
