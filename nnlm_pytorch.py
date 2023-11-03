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
from torch.utils.data import TensorDataset, DataLoader, dataset



def tokenize_and_split(raw_text_iter: dataset.IterableDataset, total_samples: int) -> Tensor:
    """
    1. Tokenize and convert raw text into a list of tensors, where each tensor represents a sequence of tokens
    2. Eliminate empty tensors and concatenate into a single tensor
    3. Split the single tensor into samples, where each sample represents a sequence of tokens
    """
    
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    data = torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    length_of_sequence = data.shape[0] // total_samples 
    data = data[:length_of_sequence * total_samples]                       
    data = data.view(total_samples, length_of_sequence)   
           
    return data

def create_dataloader_from_text_data(text_data_tensor: Tensor, batch_size: int) -> DataLoader:
    dataset = TensorDataset(text_data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader 

                     
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork).__init__()
        """
        y = b + Wx + U * tanh(d + Hx)
        """
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.h = nn.Linear(n_steps * embedding_dim, n_hidden, bias=False)
        self.d = nn.Parameter(torch.ones(n_hidden))
        self.tanh = nn.Tanh()
        self.u = nn.Linear(n_hidden, vocab_size, bias=False)
        self.w = nn.Linear(n_steps * embedding_dim, vocab_size, bias=False)
        self.b = nn.Parameter(torch.ones(vocab_size))
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(-1, n_steps * embedding_dim)     # need to check this
        tanh = self.tanh(self.d + self.h(x))
        y = self.b + self.w(x) + self.u(tanh)
        return y
    

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_dim = 1000
    total_samples = 10000
    n_steps = 5
    n_hidden = 10
    batch_size = 64

    data_for_dict = WikiText2(split='train')    
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, data_for_dict), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    vocab_size = len(vocab)
    
    train_iter, val_iter, test_iter = WikiText2()
    train_data = tokenize_and_split(train_iter, total_samples)
    #val_data = tokenize_and_split(val_iter, total_samples)
    #test_data = tokenize_and_split(test_iter, total_samples)
    train_dataloader = create_dataloader_from_text_data(train_data, batch_size)
    '''
    Check the output of the dataloader

    for batch in train_dataloader:
        print(batch[0][:, -1])
        print(batch[0][:, -1].view(-1, 1))
        break
    '''
    # Training
    model = NeuralNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1000):
        for batch in train_dataloader:
            input_batch, target_batch = batch[0][:, :-1], batch[0][:, -1]
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            output = model(input_batch)
            loss = criterion(output, target_batch)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    # Predict

    # Test
    

        
    
    
