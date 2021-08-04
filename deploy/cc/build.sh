###
 # @Author: your name
 # @Date: 2021-08-03 19:33:05
 # @LastEditTime: 2021-08-04 17:26:58
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: \deploy\tfversion\build.sh
### 

gcc -c ./fft2d/fftsg2d.c -o fftsg2d.o
gcc -c ./fft2d/fftsg.c -o fftsg.o

g++ -std=c++11 demo.cc spectrogram.cc mfcc.cc mfcc_mel_filterbank.cc mfcc_dct.cc fftsg2d.o fftsg.o -o demo -I ./  -I./fft2d/