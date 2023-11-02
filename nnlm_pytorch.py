# Description: Neural Network Language Model in PyTorch
# conda install -c anaconda chardet
# conda install -c conda-forge portalocker
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch import nn, Tensor
from torch.utils.data import dataset



def tokenize_and_batch(raw_text_iter: dataset.IterableDataset, batch_size) -> Tensor:
    # Tokenize and convert raw text into a list of tensors,
    # where each tensor represents a sequence of tokens
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    # Eliminate empty tensors and concatenate into a single tensor
    data = torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    num_batches = data.shape[0] // batch_size 
    data = data[:num_batches * batch_size]                       
    data = data.view(batch_size, num_batches)   
           
    return data

def get_batch(data, seq_len, idx):
    src = data[:, idx:idx+seq_len]                   
    target = data[:, idx+1:idx+seq_len+1]             
    return src, target

                     
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_for_dict = WikiText2(split='train')    
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, data_for_dict), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    
    train_iter, val_iter, test_iter = WikiText2()
    train_data = tokenize_and_batch(train_iter, batch_size=32)
    val_data = tokenize_and_batch(val_iter, batch_size=32)
    test_data = tokenize_and_batch(test_iter, batch_size=32)
    
    
