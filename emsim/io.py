import struct
import numpy as np


# little-endian uint16
__dtype = np.dtype("<H").newbyteorder("<")
# You have to set the byte order twice for it to say it's little?
assert __dtype.byteorder == "<"
__header_len_in_bytes = 4


def read_images(datfile: str, n_images: int):
    """
    Get the first n_images images from a scan datafile.
    Each "image" is the difference between two 512x512 "pre"-images
    read in succession. Each image is median subtracted.
    """

    num_values = 512 * 512 * 2 * n_images

    data = np.fromfile(
        datfile, dtype=__dtype, offset=__header_len_in_bytes, count=num_values
    )
    data = data.astype(int)

    data = data.reshape(-1, 512, 512)

    img1 = data[::2]
    img2 = data[1::2]

    imgs = img2 - img1
    med = np.median(imgs, axis=0)
    imgs = imgs - med

    return imgs


def bytes_from_file(filename, chunksize=2):
    """return an iterator over a binary file"""
    with open(filename, "rb") as f:
        while True:
            chunk = f.read(chunksize)
            if chunk:
                yield struct.unpack("<H", chunk)[0]
            else:
                break


def read_images_old(datfile, n_images):
    """
    Get the first n_images images from a scan datafile.
    Each "image" is the difference between two 512x512 "pre"-images
    read in succession. Each image is median subtracted.
    """

    # Create a reader.
    freader = iter(bytes_from_file(datfile))

    # Read 4-byte header.
    for i in range(2):
        next(freader)

    # Read the images.
    imgs = []
    for ni in range(n_images):

        img1 = []
        for i in range(512 * 512):
            img1.append(next(freader))

        img2 = []
        for i in range(512 * 512):
            img2.append(next(freader))

        imgs.append(
            np.array(img2).reshape([512, 512]) - np.array(img1).reshape([512, 512])
        )

    # Return the final image array in numpy format.
    imgs = np.array(imgs)
    imgs = imgs - np.median(imgs, axis=0)

    return imgs
