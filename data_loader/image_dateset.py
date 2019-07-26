import numpy as np
from cv2 import cv2 
import csv
import random


"""
def parse_csv(line):
    example_defaults = [[0.], [0.], [0.], [0.], [0]]
    parsed_line = tf.decode_csv(line, example_defaults)
    features = tf.reshape(parsed_line[:-1], shape=(4,))
    label = tf.reshape(parsed_line[-1], shape=())
    return features, label

if __name__ == "__main__":

    train_dataset_csv = "./test.csv"

    train_dataset = tf.data.TextLineDataset(train_dataset_csv)

    train_dataset = train_dataset.skip(1)                    # skip the first header row 
    train_dataset = train_dataset.map(parse_csv)             # parse each row
    train_dataset = train_dataset.shuffle(buffer_size=1000)  # randomize
    train_dataset = train_dataset.batch(32)                  # batch

    # view a single example entry from a batch
    next_train_data = train_dataset.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        #print(type(next_element), next_element)
        features, label = sess.run(next_train_data)
        print("features:\n", features)
        print("labels:\n", label)
"""

class DataGenerator:
    def __init__(self, config):
        self.config = config

        self.dataset = []
        with open(config.dataset, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.dataset.append(row)

        if len(self.dataset) == 0:
            raise Exception("get no data from file "+config.dataset)


    """
    def next_batch(self, batch_size):
        idx = np.random.choice(len(self.dataset), batch_size)

        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.y[idx]
    """

    def batches(self, batch_size):
        indexs = list(range(len(self.dataset)))
        random.shuffle(indexs)

        indexs_batches = [indexs[x:x+batch_size] for x in range(0,len(indexs),batch_size)]
    
        for indexs_batch in indexs_batches:
            x_batch = np.zeros([len(indexs_batch)] + self.config.image_shape, np.float32)
            y_batch = np.zeros(len(indexs_batch), np.int32)

            for i, index in enumerate(indexs_batch):
                # get image data
                filename = self.dataset[index][0]
                label = self.dataset[index][1]
                img = cv2.imread(filename)

                # resize
                if self.config.resize:
                    img = cv2.resize(img, tuple(self.config.image_shape[0:2]))
                    # img = cv2.resize(img, self.config.image_shape[0:2], interpolation=cv2.INTER_CUBIC)

                x_batch[i] = img.astype(np.float32)
                y_batch[i] = label
            yield x_batch, y_batch
            
            
if __name__ == "__main__":
    from utils.config import process_config
    from utils.utils import get_args

    args = get_args()
    config = process_config(args.config)

    generator = DataGenerator(config)

    for x_batch, y_batch in generator.batches(5):
        print(x_batch.shape)
        print(y_batch.shape)
        cv2.imshow("window", x_batch[0].astype(np.uint8))
        cv2.waitKey(0)
        
            