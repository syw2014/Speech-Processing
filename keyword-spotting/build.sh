if [ ! -d build ]; then
    mkdir build
fi


cd build
rm -rf *


#export CC="/home/soft/gcc/gcc-7.3.0/objdir/bin/gcc"
#export CXX="/home/soft/gcc/gcc-7.3.0/objdir/bin/g++"
#export CC="/home/platform/gcc_cmake/gcc-5.4.0/objdir/bin/gcc"
#export CXX="/home/platform/gcc_cmake/gcc-5.4.0/objdir/bin/g++"

#export CC="/mnt/algo/algo_temp/algo/ml_framework/tensorflow/tf-2.2.0/rh/devtoolset-7/root/bin/gcc"
#export CXX="/mnt/algo/algo_temp/algo/ml_framework/tensorflow/tf-2.2.0/rh/devtoolset-7/root/bin/g++"
export CC="/usr/local/bin/gcc"
export CXX="/usr/local/bin/g++"


if [ $1 == 1 ]; then
    cmake .. -DCMAKE_BUILD_TYPE=DEBUG || exit 1
elif [ $1 == 2 ]; then
    cmake .. -DCMAKE_BUILD_TYPE=RELEASE || exit 1
fi


make -j4
