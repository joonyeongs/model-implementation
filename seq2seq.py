# Description: Seq2seq Model in PyTorch


import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch import nn, Tensor
from torch.utils.data import DataLoader, dataset
from torch.nn.utils.rnn import pad_sequence


def create_vocab_from_text_data() -> tuple:
    sentences = {src_language: [], tgt_language: []}
    data_for_dict = Multi30k(split="train", language_pair=language_pair) 
    for src_sentence, tgt_sentence in data_for_dict:
        sentences[src_language].append(src_sentence)
        sentences[tgt_language].append(tgt_sentence)

    tokenizer = {}
    tokenizer[src_language] = get_tokenizer('spacy', language='en_core_web_sm')
    tokenizer[tgt_language] = get_tokenizer('spacy', language='de_core_news_sm')

    vocab = {}
    vocab_size = []
    for lang in language_pair:
        vocab[lang] = build_vocab_from_iterator(map(tokenizer[lang], sentences[lang]), specials=special_symbols)
        vocab[lang].set_default_index(vocab[lang]['<unk>'])
        vocab_size.append(len(vocab[lang]))

    return vocab, vocab_size, tokenizer
    
def tokenize_text_data(raw_text_iter: str, language: str) -> list:
    tokens = tokenizer[language](raw_text_iter)

    return tokens

def map_token_to_index(tokens: list, language: str) -> list:
    if language == tgt_language:
        tokens = ['<sos>'] + tokens + ['<eos>']

    indices = [vocab[language][token] for token in tokens]

    return indices
    

def transform_text_data_to_tensor(text_data: Tensor, language: str) -> Tensor:
    tokens = tokenize_text_data(text_data, language)
    indices = map_token_to_index(tokens, language)
    tensor = torch.tensor(indices, dtype=torch.long)

    return tensor

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(transform_text_data_to_tensor(src_sample, src_language))
        tgt_batch.append(transform_text_data_to_tensor(tgt_sample, tgt_language))

    src_batch = pad_sequence(src_batch, padding_value=pad_idx).transpose(0, 1)
    tgt_batch = pad_sequence(tgt_batch, padding_value=pad_idx).transpose(0, 1)

    return src_batch, tgt_batch

def create_padding_mask(src, tgt):
    src_padding_mask = (src != pad_idx)
    tgt_padding_mask = (tgt != pad_idx)
    return src_padding_mask, tgt_padding_mask   

def train():
    train_loss = 0
    for batch in train_dataloader:
        src_batch, tgt_batch = batch
        src_padding_mask, tgt_padding_mask = create_padding_mask(src_batch, tgt_batch)

        optimizer.zero_grad()
        output = model(src_batch)
        train_loss = criterion(output, tgt_batch)
        train_loss.backward()
        optimizer.step()
    
    return train_loss

def test():
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0

    for batch in test_dataloader:
        source_batch, target_batch = batch

        output = model(source_batch)
        loss = criterion(output, target_batch)
        test_loss += loss.item()
        correct += (output.argmax(1) == target_batch).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    return test_loss, correct

class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim)

    def forward(self, x):
        x = self.embedding(x)

        return y
        

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 200
    hidden_dim = 100
    num_layers = 3
    length_of_sequence = 20
    batch_size = 64
    src_language = "en"
    tgt_language = "de"
    language_pair = (src_language, tgt_language)
    special_symbols = ['<unk>', '<sos>', '<eos>', '<pad>']
    unk_idx, sos_idx, eos_idx, pad_idx = 0, 1, 2, 3

    vocab, vocab_size, tokenizer = create_vocab_from_text_data()
    
    train_data, test_data = Multi30k(split=('train', 'test'), language_pair=language_pair)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size)

    for batch in train_dataloader:
        src_batch, tgt_batch = batch
        sentence = src_batch[3]
        print(sentence)
    
        indices = [vocab[src_language].get_itos()[idx] for idx in sentence]
        print(indices)
        break
'''
    # Training
    model = Seq2Seq().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(100):
        train_loss = train()

        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(train_loss))

    # Test
    
    with torch.no_grad():
        test_loss, correct = test()

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")'''
    

        
    
    
