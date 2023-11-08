# Description: Neural Network Language Model in PyTorch
# conda install -c anaconda chardet
# conda install -c conda-forge portalocker
# conda install -c pytorch torchtext
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch import nn, Tensor
from torch.utils.data import TensorDataset, DataLoader, dataset

def create_vocab_from_text_data() -> tuple:
    data_for_dict = WikiText2(split='train')    
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, data_for_dict), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    vocab_size = len(vocab)

    return vocab, vocab_size, tokenizer
    

def tokenize_and_split(raw_text_iter: dataset.IterableDataset, length_of_sequence: int) -> Tensor:
    """
    1. Tokenize and convert raw text into a list of tensors, where each tensor represents a sequence of tokens
    2. Eliminate empty tensors and concatenate into a single tensor
    3. Split the single tensor into samples, where each sample represents a sequence of tokens
    """
    
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    data = torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    total_samples = data.shape[0] // length_of_sequence 
    data = data[:length_of_sequence * total_samples]                       
    data = data.view(total_samples, length_of_sequence)   
           
    return data

def create_dataloader_from_text_data(text_data_tensor: Tensor, batch_size: int) -> DataLoader:
    dataset = TensorDataset(text_data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def split_batch_into_input_and_target(batch: list) -> tuple:
    input_batch, target_batch = batch[0][:, :-1], batch[0][:, -1]
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    return input_batch, target_batch

def train():
    for batch in train_dataloader:
        input_batch, target_batch = split_batch_into_input_and_target(batch)

        optimizer.zero_grad()
        output = model(input_batch)
        train_loss = criterion(output, target_batch)
        train_loss.backward()
        optimizer.step()


def test():
    for batch in test_dataloader:
        input_batch, target_batch = split_batch_into_input_and_target(batch)

        output = model(input_batch)
        loss = criterion(output, target_batch)
        test_loss += loss.item()
        correct += (output.argmax(1) == target_batch).type(torch.float).sum().item()

class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()
        self.n_steps = length_of_sequence - 1
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.Wh = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        hidden_states, _ = self.rnn(x)
        hidden_states = hidden_states[:, -1, :]
        y = self.Wh(hidden_states)
        return y
        

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 200
    hidden_dim = 100
    num_layers = 3
    length_of_sequence = 20
    batch_size = 64

    vocab, vocab_size, tokenizer = create_vocab_from_text_data()
    
    train_iter, test_iter = WikiText2(split='train'), WikiText2(split='test')
    train_data = tokenize_and_split(train_iter, length_of_sequence)
    test_data = tokenize_and_split(test_iter, length_of_sequence)
    train_data, test_data = train_data[:100, :], test_data[:100, :]
    train_dataloader = create_dataloader_from_text_data(train_data, batch_size)
    test_dataloader = create_dataloader_from_text_data(test_data, batch_size)
   

    
    # Training
    model = TextRNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train()
    '''
    for epoch in range(1000):
        train_loss = 0
        train()

        if (epoch + 1) % 100 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(train_loss))

    # Test
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        test()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    '''

        
    
    
