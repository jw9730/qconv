import sys
import numpy as np
import cv2
import time
import struct
from array import array
import convolution


def image_object_detection(input_bin, kernel_bin):
    # read input image
    with open(input_bin, "rb") as h:
        file_content = h.read()
        N, H, W, C = struct.unpack("iiii", file_content[:4 * 4])
        frame = np.asarray(struct.unpack("f"*N*H*W*C, file_content[4 * 4:])).reshape((N, H, W, C))
        if N != 1: raise ValueError
        frame = np.ascontiguousarray(frame.transpose((0, 2, 1, 3)))

    with open(kernel_bin, "rb") as h:
        file_content = h.read()
        KH, KW, OC, IC = struct.unpack("iiii", file_content[:4 * 4])
        kernel = np.asarray(struct.unpack("f" * KH * KW * OC * IC, file_content[4 * 4:])).reshape((KH, KW, OC, IC))
        kernel = np.ascontiguousarray(kernel.transpose((1, 0, 3, 2)))

    y2t = convolution.CONV([N, W, H, C], kernel)

    tout = y2t.inference(frame)
    tout = np.ascontiguousarray(tout.transpose((0, 2, 1, 3)))

    with open("./output_tensor.bin", "wb") as h:
        np.asarray([N, H, W, OC]).astype('int32').tofile(h)
        tout.astype('float32').tofile(h)

    print("created reference file")


def main():
    if len(sys.argv) < 3:
        print ("Usage: python3 __init__.py [input.bin] [kernel.bin]")
        sys.exit()

    input_bin = sys.argv[1]
    kernel_bin = sys.argv[2]
    image_object_detection(input_bin, kernel_bin)

if __name__ == "__main__":
    main()
