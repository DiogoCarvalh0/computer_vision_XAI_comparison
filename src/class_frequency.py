"""
Gives the frequency of each class.
arguments:
    XML_DIR: Path to the xml files
"""

import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm
from pprint import pprint


XML_DIR = './../data/test/annotations/'

def add_object_to_count(count, object_name):
    try:
        count[object_name] += 1
    except:
        count[object_name] = 1

    return count


def main():
    count = {}

    for xml_file in tqdm(glob.glob(f'{XML_DIR}*.xml')):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for member in root.findall('object'):
            object_name = member.find('name').text

            count = add_object_to_count(count, object_name)

    pprint(count)


if __name__ == '__main__':
    main()