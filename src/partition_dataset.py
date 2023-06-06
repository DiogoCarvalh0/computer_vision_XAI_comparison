"""
Partition dataset of images into training and testing sets
"""
import os
import re
from shutil import copyfile
import math
import random


IMAGE_DIR = './../data/images/'
OUTPUT_DIR = './../data/'
RATIO = 0.2
COPY_XML = True


def iterate_dir(source=os.getcwd(), dest=None, ratio=0.1, copy_xml=True):
    """
    :param source: Path to the folder where the image dataset is stored. If not specified, the CWD will be used.
    :param dest: Path to the output folder where the train and test dirs should be created.
                 Defaults to the same directory as IMAGEDIR.
    :param ratio: The ratio of the number of test images over the total number of images. The default is 0.1.',
    :param copy_xml: Set this flag if you want the xml annotation files to be processed and copied over.
    """

    if dest is None:
        dest = source

    source = source.replace('\\', '/')
    dest = dest.replace('\\', '/')
    train_dir = os.path.join(dest, 'train')
    test_dir = os.path.join(dest, 'test')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    images = [f for f in os.listdir(source) if re.search(r'([a-zA-Z0-9\s_\\.\-():])+(?i)(.jpg|.jpeg|.png)$', f)]

    num_images = len(images)
    num_test_images = math.ceil(ratio*num_images)

    for _ in range(num_test_images):
        idx = random.randint(0, len(images)-1)
        filename = images[idx]
        copyfile(os.path.join(source, filename),
                 os.path.join(test_dir, filename))
        if copy_xml:
            xml_filename = os.path.splitext(filename)[0]+'.xml'
            copyfile(os.path.join(source, xml_filename),
                     os.path.join(test_dir, xml_filename))
        images.remove(images[idx])

    for filename in images:
        copyfile(os.path.join(source, filename),
                 os.path.join(train_dir, filename))
        if copy_xml:
            xml_filename = os.path.splitext(filename)[0]+'.xml'
            copyfile(os.path.join(source, xml_filename),
                     os.path.join(train_dir, xml_filename))


def main():
    # Now we are ready to start the iteration
    iterate_dir(IMAGE_DIR, OUTPUT_DIR, RATIO, COPY_XML)


if __name__ == '__main__':
    main()