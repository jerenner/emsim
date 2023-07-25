import time
import glob
import multiprocessing
import struct
import os
import functools
import re

import h5py
import numpy as np


# little-endian uint16
__dtype = np.dtype("<H").newbyteorder("<")
# You have to set the byte order twice for it to say it's little?
assert __dtype.byteorder == "<"
__header_len_in_bytes = 4


def read_images(datfile: str, n_images=2048, subtract_median=True):
    """
    Get the first n_images images from a scan datafile.
    Each "image" is the difference between two 512x512 "pre"-images
    read in succession. Each image is median subtracted by default.
    """

    num_values = 512 * 512 * 2 * n_images

    data = np.fromfile(
        datfile, dtype=__dtype, offset=__header_len_in_bytes, count=num_values
    )
    data = data.astype(np.int32)

    data = data.reshape(-1, 512, 512)

    img1 = data[::2]
    img2 = data[1::2]

    imgs = img2 - img1
    if subtract_median:
        del img1, img2, data
        med = np.median(imgs, axis=0)
        imgs = imgs - med

    return imgs


def process_single_datfile(datfile, n_images, output_h5_file=None, threshold=100, subtract_median=True):
    """
    Loads n_images images from a scan datfile, and writes them to an h5 file.
    These images are written to an hd5file. Also, a set of images binarized by
    testing of each pixel has a value above the specified `threshold` is
    produced and also saved to the hdf5 file.

    Args:
        datfile (str): Path to the datfile
        n_images (int): Number of images in the datfile
        output_h5_file (str, optional): Path of h5 file to write to. If None,
            uses the name of the datfile with the extension changed to hdf5.
        threshold (int, optional): Value to binarize over. Defaults to 100.
        subtract_median (bool, optional): If True, the median value at each
            pixel over the `n_images` images is subtracted from the saved images.
            Defaults to True.
    """
    imgs = read_images(datfile, n_images=n_images, subtract_median=subtract_median)

    imgs_processed = np.mean(imgs, axis=0)

    imgs_thresholded = imgs > threshold
    imgs_thresholded_processed = np.sum(imgs_thresholded, axis=0)

    datfile_number = re.search("(?<=_)\d+(?=.dat)", datfile)[0]

    if output_h5_file is None:
        data_dir, scan_dir = os.path.split(os.path.dirname(datfile))
        output_h5_file = os.path.join(data_dir, scan_dir + ".hdf5")

    with h5py.File(output_h5_file, "a") as h5file:
        grp = h5file.create_group(datfile_number)
        grp.create_dataset("imgs_processed", data=imgs_processed)
        grp.create_dataset("imgs_thresholded_processed", data=imgs_thresholded_processed)

    return output_h5_file


def load_and_process_directory(directory, threshold=100, n_images=2048):
    directory = os.path.join(directory, "")
    datfiles = glob.glob(directory + "*.dat")

    # with multiprocessing.Pool(processes=nproc) as pool:
    #     imgs = pool.map(
    #         functools.partial(read_images, n_images=n_images, subtract_median=False),
    #         datfiles,
    #     )

    all_imgs = [
        read_images(datfile, 2048, n_images=n_images) for datfile in datfiles]

    imgs_processed = np.stack([np.mean(imgs, axis=0) for imgs in all_imgs], axis=0)

    imgs_thresholded = [imgs > threshold for imgs in all_imgs]
    imgs_processed_thresholded = [np.sum(imgs, axis=0) for imgs in imgs_thresholded]
    imgs_processed_thresholded = np.stack(imgs_processed_thresholded, axis=0)

    return imgs_processed, imgs_processed_thresholded


def pad_stack_imgs_and_get_mask(imgs):
    shapes = np.stack([img.shape for img in imgs], axis=0)
    max_shape = np.max(shapes, axis=0)
    pad_size = shapes - max_shape
    to_pad = [[(0, axis_len) for axis_len in pad_len] for pad_len in pad_size]
    padded = np.stack([np.pad(img, pad) for img, pad in zip(imgs, to_pad)], axis=0)

    mask = np.zeros_like(padded, dtype=bool)
    max_n_images = max_shape[0]
    for i, mask_len in enumerate(pad_size):
        mask[i, max_n_images - mask_len :] = True

    padded_masked = np.ma.masked_array(padded, mask)

    return padded_masked


def load_with_timer(datfile, n_images):
    start = time.process_time()
    imgs = read_images(datfile, n_images)
    elapsed = time.process_time() - start
    print(f"Read {datfile} in {elapsed} s")
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
