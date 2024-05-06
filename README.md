# pyFFTps
pyFFTps can be used to do mass assign for simulation particles and calculate the power spectrum.
The basic code is from the [Pylians3](https://github.com/franciscovillaescusa/Pylians3/tree/master).
We modified the code to support the interlacing technology and do the shotnoise substration from the density field.
Therefore, the calculated power spectrum is accurate toward even $\bold{Nyquist}$ frequency!
For more details, please read the [Wang & Yu 2024](https://arxiv.org/pdf/2403.13561).

## Installation
You can just modify the python path in the build.sh and run
```
./build.sh
```

This package is dependent only on the [pyFFTW](https://github.com/pyFFTW/pyFFTW/tree/master) which provide the thread-parallel FFT algorithm.

## Usage

Please view the test directory.

## Reference

Please cite the [Wang & Yu 2024](https://arxiv.org/pdf/2403.13561) if you find this code useful in your research.

## Contributions
We welcome all contributions to pyFFTps via Pull Requests. 
Let us know about any issues or questions about pyFFTps.


