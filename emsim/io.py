import struct


def bytes_from_file(filename, chunksize=2):
    """return an iterator over a binary file"""
    with open(filename, "rb") as f:
        while True:
            chunk = f.read(chunksize)
            if chunk:
                yield struct.unpack("<H", chunk)[0]
            else:
                break
