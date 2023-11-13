export ROOT_DIR=$PWD

rm -rf build 
rm -rf hemul.egg*

cd $ROOT_DIR/HEAAN/lib/
make clean && make  -j 4

cd $ROOT_DIR
pip install -e .
