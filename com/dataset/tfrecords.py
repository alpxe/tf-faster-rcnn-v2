# -*- coding: utf-8 -*-
import os
import cv2
import pandas as pd
import tensorflow as tf
import com.dataset.xml2csv as xml2csv

from collections import namedtuple, OrderedDict
from com.util import dataset_util


class Records():
    def __init__(self):
        self.path = ""
        pass

    def class_text_to_int(self, row_label):
        """
        文本标签标记分类
        :param row_label:
        :return:
        """
        if row_label == 'JJY':
            return 1
        else:
            None
        pass

    def __imageProcess(self, image):
        h, w = image.shape[:2]

        max = 512  # 最大边

        pro = h / w
        if pro > 0:
            resize_h = max
            resize_w = max * w / h
        else:
            resize_w = max
            resize_h = max * h / w

        return cv2.resize(image, (int(resize_w), int(resize_h)), interpolation=cv2.INTER_AREA)

    def __format(self, record):
        fs = {
            "height": tf.FixedLenFeature((), tf.int64),
            "width": tf.FixedLenFeature((), tf.int64),
            "filename": tf.FixedLenFeature((), tf.string),
            "image": tf.FixedLenFeature((), tf.string),  # img 一维展开
            "tag/xmin": tf.FixedLenFeature((), tf.float32),
            "tag/xmax": tf.FixedLenFeature((), tf.float32),
            "tag/ymin": tf.FixedLenFeature((), tf.float32),
            "tag/ymax": tf.FixedLenFeature((), tf.float32),
            "class/text": tf.FixedLenFeature((), tf.string),
            "class/label": tf.FixedLenFeature((), tf.int64)
        }
        fats = tf.parse_single_example(record, features=fs)

        height = tf.cast(tf.squeeze(fats["height"]), tf.int32)
        width = tf.cast(tf.squeeze(fats["width"]), tf.int32)

        image = tf.image.decode_image(fats["image"], dtype=tf.float32)
        image = tf.reshape(image, shape=[height, width, 3])

        data = {
            "height": height,
            "width": width,
            "filename": fats["filename"],
            "image": image,
            "xmin": fats["tag/xmin"],
            "xmax": fats["tag/xmax"],
            "ymin": fats["tag/ymin"],
            "ymax": fats["tag/ymax"],
            "text": fats["class/text"],
            "label": fats["class/label"]
        }
        return data

    def create_tf_example(self, group, path):
        img_path = os.path.join(path, group.filename)  # 图片的完整路径
        img = cv2.imread(img_path)

        img = self.__imageProcess(img)  # 修正尺寸

        # 将 BGR转RGB
        # tf.image.decode_image 很奇怪，又会反过来变成BGR  BGR由imshow现实会正常
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # height, width, depth = img.shape  # 获取图片宽高深
        filename = group.filename.encode("utf8")  # bytes

        # 一个小组的内容
        for index, row in group.object.iterrows():
            width = row['width']
            height = row['height']
            xmins = [row['xmin'] / width]
            xmaxs = [row['xmax'] / width]
            ymins = [row['ymin'] / height]
            ymaxs = [row['ymax'] / height]  # 对于宽高所在位置百分比

            classes_text = [row['class'].encode("utf8")]
            classes = [self.class_text_to_int(row['class'])]
            pass

        # do nothing 此处可能需要调整图片尺寸

        height, width, depth = img.shape  # 获取图片宽高深

        encoded_jpg = cv2.imencode('.jpeg', img)[1].tostring()  # opencv  mat -> bytes

        # 创建一个 train.Example
        fs = tf.train.Features(feature={
            "height": dataset_util.int64_feature(height),
            "width": dataset_util.int64_feature(width),
            "filename": dataset_util.bytes_feature(filename),
            "image": dataset_util.bytes_feature(encoded_jpg),
            "tag/xmin": dataset_util.float_list_feature(xmins),
            "tag/xmax": dataset_util.float_list_feature(xmaxs),
            "tag/ymin": dataset_util.float_list_feature(ymins),
            "tag/ymax": dataset_util.float_list_feature(ymaxs),
            "class/text": dataset_util.bytes_list_feature(classes_text),
            "class/label": dataset_util.int64_list_feature(classes)
        })
        return tf.train.Example(features=fs)

    def __split(self, df, group):
        data = namedtuple("data", ["filename", "object"])
        """
        返回一个具名元组子类 typename，其中参数的意义如下：

        typename：元组名称
        field_names: 元组中元素的名称
        rename: 如果元素名称中含有 python 的关键字，则必须设置为 rename=True
        verbose: 默认就好
        """

        gb = df.groupby(group)  # [17 rows x 8 columns] 以名字分组

        # print(gb.groups) # {filename: columns data}  gb.get_group(filename)>>获取data

        return [data(key, gb.get_group(key)) for key in gb.groups.keys()]

    def xml2csv(self, base_image_path, csv_path):
        """
        xml 转 csv
        :param base_image_path: 图片基础路径
        :param csv_path: csv的存储位置
        :return:
        """
        # 将xml数据集转成csv
        xml2csv.run(base_image_path, csv_path)
        pass

    def generate(self, csv_path, tf_path, base_image_path):
        """
        生成 tfrecords 数据集
        :param csv_path: csv文件的路径
        :param tf_path: 生成 tfrecords文件的保存路径
        :param base_image_path: 图片的基础路径
        :return:
        """
        examples = pd.read_csv(csv_path)
        grouped = self.__split(examples, 'filename')  # [{filename:xxx,object:yyy},...] 的对象

        writer = tf.python_io.TFRecordWriter(tf_path)
        for i, group in zip(range(len(grouped)), grouped):
            tf_example = self.create_tf_example(group, base_image_path)
            writer.write(tf_example.SerializeToString())  # SerializeToString 写入

            # sys.stdout.write("\rWrite progress： {0:.2f}%".format((i + 1) / len(grouped) * 100))
            pass
        writer.close()

        print('\nSuccessfully created the TFRecords')
        pass

    def extract(self, path):
        """
        提取数据集
        :param path:
        :return:
        """

        # 1.>>>>得到数据
        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.repeat()

        dataset = dataset.map(self.__format)

        dataset = dataset.batch(1)  # 因为batch出的尺寸必须一致

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    pass
