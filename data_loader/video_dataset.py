import numpy as np
from cv2 import cv2 
import csv
import random

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



    def batches(self, batch_size):
        indexs = list(range(len(self.dataset)))
        random.shuffle(indexs)

        indexs_batches = [indexs[x:x+batch_size] for x in range(0,len(indexs),batch_size)]
    
        for indexs_batch in indexs_batches:
            x_batch = np.zeros([len(indexs_batch), self.config.max_num_frame] + self.config.frame_shape, np.float32)
            y_batch = np.zeros(len(indexs_batch), np.int32)

            for i, index in enumerate(indexs_batch):
                # open video
                filename = self.dataset[index][0]
                label = self.dataset[index][1]
                video = cv2.VideoCapture(filename)
                assert(video.isOpened())

                for frame_index in range(self.config.max_num_frame):
                    # read frame
                    ret, frame = video.read()
                    if not ret:
                        # fill the rest frame with the last frame
                        assert(frame_index>0)
                        x_batch[i, frame_index:] = x_batch[i, frame_index-1]

                        break

                    # resize
                    if self.config.resize:
                        frame = cv2.resize(frame, tuple(self.config.frame_shape[0:2]))
                    
                    x_batch[i,frame_index] = frame.astype(np.float32)
                    y_batch[i] = label

                
                
                print()
                video.release()
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

        cv2.imshow("window", x_batch[0, 20].astype(np.uint8))
        cv2.waitKey(5)
        
            