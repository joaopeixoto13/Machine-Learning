import numpy as np

# airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
# Example: if is a dog
# 0, 0, 0, 0, 0, 1, 0, 0, 0, 0
def one_hot_encode(vec, vals=10):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out

class CifarHelper():

    def __init__(self, data_batch1, data_batch2, data_batch3, data_batch4, data_batch5, test_batch):
        self.i = 0

        self.all_train_batches = [data_batch1, data_batch2, data_batch3, data_batch4, data_batch5]
        self.test_batch = [test_batch]

        self.training_images = None
        self.training_labels = None

        self.test_images = None
        self.test_labels = None


    def set_up_images(self):

        print("Setting Up Training Images and Labels")

        self.training_images = np.vstack([d[b'data'] for d in self.all_train_batches])
        train_len = len(self.training_images)

        # 0-1 all images 32px to 32px in RGB
        self.training_images = self.training_images.reshape(train_len, 3, 32, 32).transpose(0, 2, 3, 1)/255
        self.training_labels = one_hot_encode(np.hstack([d[b'labels'] for d in self.all_train_batches]), 10)

        print("Setting Up Test Images and Labels")

        self.test_images = np.vstack([d[b'data'] for d in self.test_batch])
        test_len = len(self.test_images)

        self.test_images = self.test_images.reshape(test_len, 3, 32, 32).transpose(0, 2, 3, 1)/255
        self.test_labels = one_hot_encode(np.hstack([d[b'labels'] for d in self.test_batch]), 10)

    def next_batch(self, batch_size):
        x = self.training_images[self.i:self.i + batch_size].reshape(100, 32, 32, 3)
        y = self.training_labels[self.i:self.i + batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y