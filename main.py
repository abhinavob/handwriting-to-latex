from dataset import HMEDataset

dataset = HMEDataset("../hme100k", "../hme100k/train.txt")

image, label = dataset[0]

print(image.shape)
print(label)