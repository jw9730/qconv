import sys
import numpy as np
import struct

def main():
    if len(sys.argv) < 3:
        print ("Usage: python3 test.py [ref_output.bin] [test_output.bin]")
        sys.exit()

    ref_output_bin = sys.argv[1]
    test_output_bin = sys.argv[2]

    with open(ref_output_bin, "rb") as h:
        file_content = h.read()
        N, H, W, C = struct.unpack("iiii", file_content[:4 * 4])
        #print("parsed shape {}".format([N,H,W,C]))
        ref_output = np.asarray(struct.unpack("f" * N * H * W * C, file_content[4 * 4:])).reshape((N, H, W, C))
        if N != 1: raise ValueError
        ref_output = np.ascontiguousarray(ref_output.transpose((0, 2, 1, 3)))

    with open(test_output_bin, "rb") as h:
        file_content = h.read()
        N2, H2, W2, C2 = struct.unpack("iiii", file_content[:4 * 4])
        #print("parsed shape {}".format([N2,H2,W2,C2]))
        test_output = np.asarray(struct.unpack("f" * N2 * H2 * W2 * C2, file_content[4 * 4:])).reshape((N2, H2, W2, C2))
        test_output = np.ascontiguousarray(test_output.transpose((0, 2, 1, 3)))


    # test routine
    assert (N==N2 and H==H2 and W==W2 and C==C2)
    print("correctness check: mean absolute error {}".format(abs(ref_output - test_output).mean()))

if __name__ == "__main__":
    main()