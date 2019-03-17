import numpy as np
import csv
import os

from .util import read_image

class SimpleDataset:
    def __init__(self, data_dir, split='trainval'):
        self.data_dir = data_dir

        ids_path = os.path.join(data_dir, 'ids.txt')
        self.ids = [id_.strip() for id_ in open(ids_path)]

        classes_path = os.path.join(data_dir, 'classes.txt')
        self.label_names = [name.strip() for name in open(classes_path)] + ['empty']
        self.class_nums = {name: index for (index, name) in enumerate(self.label_names)}

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        id_ = self.ids[i]
        bboxes = list()
        labels = list()
        anno_path = os.path.join(self.data_dir, 'annotations/{}.csv'.format(id_))
        with open(anno_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for class_name, min_y, min_x, max_y, max_x in reader:
                bboxes.append([min_y, min_x, max_y, max_x])
                labels.append(self.class_nums[class_name])
        if len(bboxes) == 0:
            bboxes = [[0, 0, 1, 1]]
            labels = [self.class_nums['empty']]
        bboxes = np.stack(bboxes).astype(np.float32)
        labels = np.stack(labels).astype(np.int32)

        img_file = os.path.join(self.data_dir, 'images/{}.jpg'.format(id_))
        img = read_image(img_file, color=True)

        return img, bboxes, labels
