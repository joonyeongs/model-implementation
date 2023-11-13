# Description: Seq2seq Model in PyTorch


import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch import nn, Tensor
from torch.utils.data import TensorDataset, DataLoader, dataset
from torch.nn.utils.rnn import pad_sequence


def create_vocab_from_text_data() -> tuple:
    sentences = {src_language: [], tgt_language: []}
    data_for_dict = Multi30k(split="train", language_pair=language_pair) 
    for src_sentence, tgt_sentence in data_for_dict:
        if len(src_sentence) > max_length or len(tgt_sentence) > max_length:
            continue
        sentences[src_language].append(src_sentence)
        sentences[tgt_language].append(tgt_sentence)

    tokenizer = {}
    tokenizer[src_language] = get_tokenizer('spacy', language='en_core_web_sm')
    tokenizer[tgt_language] = get_tokenizer('spacy', language='de_core_news_sm')

    vocab = {}
    vocab_size = {}
    for lang in language_pair:
        vocab[lang] = build_vocab_from_iterator(map(tokenizer[lang], sentences[lang]), specials=special_symbols)
        vocab[lang].set_default_index(vocab[lang]['<unk>'])
        vocab_size[lang] = len(vocab[lang])

    return vocab, vocab_size, tokenizer
    
def tokenize_text_data(raw_text_iter: str, language: str) -> list:
    tokens = tokenizer[language](raw_text_iter)
    tokens = ['<sos>'] + tokens + ['<eos>']

    return tokens

def map_token_to_index(tokens: list, language: str) -> list:
    indices = [vocab[language][token] for token in tokens]

    return indices
    

def transform_tokens_to_tensor(tokens: list, language: str) -> Tensor:
    indices = map_token_to_index(tokens, language)
    tensor = torch.tensor(indices, dtype=torch.long)

    return tensor

def longer_than_max_length(text_data: list, language: str) -> bool:
    if len(text_data) > max_length:
        return True
    else:
        return False
    
def create_dataloader_from_text_data(text_data: dataset , batch_size: int) -> DataLoader:
    src_batch, tgt_batch = [], []

    for src_sentence, tgt_sentence in text_data:
        src_sentence = tokenize_text_data(src_sentence, src_language)
        tgt_sentence = tokenize_text_data(tgt_sentence, tgt_language)
        if longer_than_max_length(src_sentence, src_language) or longer_than_max_length(tgt_sentence, tgt_language):
            continue
        src_sentence = transform_tokens_to_tensor(src_sentence, src_language)
        tgt_sentence = transform_tokens_to_tensor(tgt_sentence, tgt_language)

        src_pad_tensor, tgt_pad_tensor = (torch.full((max_length,), pad_idx) for _ in range(2))
        src_pad_tensor[:len(src_sentence)] = src_sentence
        tgt_pad_tensor[:len(tgt_sentence)] = tgt_sentence
        src_batch.append(src_pad_tensor)
        tgt_batch.append(tgt_pad_tensor)

    src_batch_tensor = torch.stack(src_batch)
    tgt_batch_tensor = torch.stack(tgt_batch)
    dataset = TensorDataset(src_batch_tensor.to(device), tgt_batch_tensor.to(device))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return dataloader


def create_padding_mask(src, tgt):
    src_padding_mask = (src != pad_idx)
    tgt_padding_mask = (tgt != pad_idx)

    src_padding_mask, tgt_padding_mask = src_padding_mask.to(device), tgt_padding_mask.to(device)

    return src_padding_mask, tgt_padding_mask   

def train():
    total_loss = 0
    for batch in train_dataloader:
        src_batch, tgt_batch = batch
        src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
        src_padding_mask, tgt_padding_mask = create_padding_mask(src_batch, tgt_batch)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        context_vector = encoder(src_batch)
        output = decoder(context_vector, tgt_batch)

        output = output[:, :-1].reshape(-1, vocab_size[tgt_language])
        target = tgt_batch[:, 1:].reshape(-1)
        tgt_padding_mask = tgt_padding_mask[:, 1:].reshape(-1)
        train_loss = criterion(output, target)
        train_loss = torch.sum(train_loss * tgt_padding_mask) / torch.sum(tgt_padding_mask)
        train_loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += train_loss.item()
    
    return total_loss / len(train_dataloader)

def test():
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss = 0

    for batch in test_dataloader:
        src_batch, tgt_batch = batch
        src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
        src_padding_mask, tgt_padding_mask = create_padding_mask(src_batch, tgt_batch)

        context_vector = encoder(src_batch)
        output = decoder(context_vector)
        loss = criterion(output, tgt_batch)
        loss = torch.sum(loss * tgt_padding_mask) / torch.sum(tgt_padding_mask)
        test_loss += loss.item()

    test_loss /= num_batches

    return test_loss

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size[src_language], input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        context_vector = hidden

        return context_vector
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size[tgt_language], input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.Wh = nn.Linear(hidden_dim, vocab_size[tgt_language])

    def forward(self, context_vector, tgt_batch=None):
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long).fill_(sos_idx).to(device)
        decoder_outputs = torch.zeros(max_length, batch_size, vocab_size[tgt_language]).to(device)
        hidden = context_vector
        cell = torch.zeros_like(hidden)

        for t in range(max_length):
            decoder_input = self.embedding(decoder_input)
            decoder_output, (hidden, cell) = self.lstm(decoder_input, (hidden, cell))
            decoder_output = self.Wh(decoder_output.view(-1, hidden_dim))
            decoder_outputs[t] = decoder_output

            if tgt_batch is not None:
                decoder_input = tgt_batch[:, t].unsqueeze(1)
            else:
                decoder_input = decoder_output.argmax(dim=1).unsqueeze(1)
        decoder_outputs = decoder_outputs.permute(1, 0, 2)       

        return decoder_outputs


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 200
    hidden_dim = 100
    num_layers = 1
    batch_size = 64
    max_length = 50
    src_language = "en"
    tgt_language = "de"
    language_pair = (src_language, tgt_language)
    special_symbols = ['<unk>', '<sos>', '<eos>', '<pad>']
    unk_idx, sos_idx, eos_idx, pad_idx = 0, 1, 2, 3

    vocab, vocab_size, tokenizer = create_vocab_from_text_data()
    
    train_data, test_data = Multi30k(split=('train', 'valid'), language_pair=language_pair)        
    train_dataloader = create_dataloader_from_text_data(train_data, batch_size)
    test_dataloader = create_dataloader_from_text_data(test_data, batch_size)
    
    
    # Training
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)
    
    for epoch in range(1000):
        train_loss = train()

        if (epoch + 1) % 100 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(train_loss))

    # Test
    
    with torch.no_grad():
        test_loss = test()

    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    

        
    
    
