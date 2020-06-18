printf "############################test routine start############################\n"
# run test for three test cases, for all convolution.c
for i in 1 2 3
do
    printf "############################testcase $i############################\n"
    # cleanup
    rm -f ./test/output_tensor.bin
    rm -f ./prob1/output_tensor.bin
    rm -f ./prob2/output_tensor.bin
    rm -f ./prob3/output_tensor.bin

    # reference
    cd ./test
    python3 ./__init__.py ../group1/$i/input_tensor.bin ../group1/$i/kernel_tensor.bin
    cd ..

    # prob1
    printf "############################prob1 test############################\n"
    cd ./prob1
    make clean; make
    ./convolution ../group1/$i/input_tensor.bin ../group1/$i/kernel_tensor.bin
    python3 ../test/test.py ./output_tensor.bin ../test/output_tensor.bin
    cd ..
    printf "############################prob1 done############################\n\n"

    # prob2
    printf "############################prob3 test############################\n"
    cd ./prob2
    make clean; make
    ./convolution ../group1/$i/input_tensor.bin ../group1/$i/kernel_tensor.bin 32
    python3 ../test/test.py ./output_tensor.bin ../test/output_tensor.bin
    ./convolution ../group1/$i/input_tensor.bin ../group1/$i/kernel_tensor.bin 16
    python3 ../test/test.py ./output_tensor.bin ../test/output_tensor.bin
    ./convolution ../group1/$i/input_tensor.bin ../group1/$i/kernel_tensor.bin 8
    python3 ../test/test.py ./output_tensor.bin ../test/output_tensor.bin
    cd ..
    printf "############################prob2 done############################\n\n"
    
    # prob3
    printf "############################prob3 test############################\n"
    cd ./prob3
    make clean; make
    ./convolution ../group1/$i/input_tensor.bin ../group1/$i/kernel_tensor.bin FP32
    python3 ../test/test.py ./output_tensor.bin ../test/output_tensor.bin
    ./convolution ../group1/$i/input_tensor.bin ../group1/$i/kernel_tensor.bin INT32
    python3 ../test/test.py ./output_tensor.bin ../test/output_tensor.bin
    ./convolution ../group1/$i/input_tensor.bin ../group1/$i/kernel_tensor.bin INT16
    python3 ../test/test.py ./output_tensor.bin ../test/output_tensor.bin
    cd ..
    printf "############################prob3 done############################\n\n"

    # prob4
    printf "############################prob4 test############################\n"
    cd ./prob4
    make clean; make
    ./convolution ../group1/$i/input_tensor.bin ../group1/$i/kernel_tensor.bin
    python3 ../test/test.py ./output_tensor.bin ../test/output_tensor.bin
    cd ..
    printf "############################prob4 done############################\n\n"
done

printf "############################ALL DONE############################\n"

# cleanup
rm -f ./test/output_tensor.bin
rm -f ./prob1/output_tensor.bin
rm -f ./prob2/output_tensor.bin
rm -f ./prob3/output_tensor.bin