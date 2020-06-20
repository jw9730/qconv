make clean; make;
./convolution ../group1/1/input_tensor.bin ../group1/1/kernel_tensor.bin FP32
./convolution ../group1/2/input_tensor.bin ../group1/2/kernel_tensor.bin FP32
./convolution ../group1/3/input_tensor.bin ../group1/3/kernel_tensor.bin FP32
./convolution ../group1/1/input_tensor.bin ../group1/1/kernel_tensor.bin INT32
./convolution ../group1/2/input_tensor.bin ../group1/2/kernel_tensor.bin INT32
./convolution ../group1/3/input_tensor.bin ../group1/3/kernel_tensor.bin INT32
./convolution ../group1/1/input_tensor.bin ../group1/1/kernel_tensor.bin INT16
./convolution ../group1/2/input_tensor.bin ../group1/2/kernel_tensor.bin INT16
./convolution ../group1/3/input_tensor.bin ../group1/3/kernel_tensor.bin INT16
