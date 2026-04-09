from dataset import HMEDataset
from tokenizer import Tokenizer

tokenizer = Tokenizer()
tokenizer.load_vocab("vocab.json")

dataset = HMEDataset("../hme100k", "../hme100k/train.txt", tokenizer)
image, tokens = dataset[0]
print(image.shape)   
print(tokens)        
