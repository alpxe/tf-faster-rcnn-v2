# -*- coding: utf-8 -*-

import glob
import pandas as pd
import xml.etree.ElementTree as ET


def __xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def run(data_path, out_path):
    """
    将xml 转成 csv
    :param data_path: 图集与xml的路径
    :param out_path: 生成csv的路径
    """

    xml_df = __xml_to_csv(data_path)
    xml_df.to_csv(out_path, index=None)

    print('Successfully converted xml to csv.')
    pass
