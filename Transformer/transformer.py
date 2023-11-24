# Description: Seq2seq+Attention Model in PyTorch


import os
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch import nn, Tensor
from torch.utils.data import TensorDataset, DataLoader, dataset
from torch.nn.utils.rnn import pad_sequence

    
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

def longer_than_max_length(tokens: list, language: str) -> bool:
    if len(tokens) > max_length:
        return True
    else:
        return False
    
def yield_tokens(tokenized_sentences: list) -> list:
    for sentence in tokenized_sentences:
        yield sentence


def create_vocab_from_text_data() -> tuple:
    data_for_dict = Multi30k(split="train", language_pair=language_pair) 
    sentences = {src_language: [], tgt_language: []}
    vocab, vocab_size = {}, {}

    for src_sentence, tgt_sentence in data_for_dict:
        src_sentence = tokenize_text_data(src_sentence, src_language)
        tgt_sentence = tokenize_text_data(tgt_sentence, tgt_language)
        if longer_than_max_length(src_sentence, src_language) or longer_than_max_length(tgt_sentence, tgt_language):
            continue
        sentences[src_language].append(src_sentence)
        sentences[tgt_language].append(tgt_sentence)


    for lang in language_pair:
        vocab[lang] = build_vocab_from_iterator(yield_tokens(sentences[lang]), specials=special_symbols, min_freq=2)
        vocab[lang].set_default_index(vocab[lang]['<unk>'])
        vocab_size[lang] = len(vocab[lang])

    return vocab, vocab_size
    

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
    dataset = TensorDataset(src_batch_tensor, tgt_batch_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    return dataloader


def create_padding_mask(src_batch: Tensor, tgt_batch: Tensor) -> tuple:
    src_padding_mask = (src_batch != pad_idx)
    tgt_padding_mask = (tgt_batch != pad_idx)

    return src_padding_mask, tgt_padding_mask   

def create_attention_mask(tgt_batch: Tensor) -> Tensor:
    batch_size, tgt_length = tgt_batch.size()
    attention_mask = torch.tril(torch.ones((tgt_length, tgt_length))).expand(batch_size, tgt_length, tgt_length)

    return attention_mask


def process_batch(batch, calculate_gradients=True):
    src_batch, tgt_batch = batch
    src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
    src_padding_mask, tgt_padding_mask = create_padding_mask(src_batch, tgt_batch)

    if calculate_gradients:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

    encoder(src_batch)
    output = decoder()

    output = output[:, :-1].reshape(-1, vocab_size[tgt_language])
    target = tgt_batch[:, 1:].reshape(-1)
    tgt_padding_mask = tgt_padding_mask[:, 1:].reshape(-1)
    loss = criterion(output, target)
    loss = torch.sum(loss * tgt_padding_mask) / torch.sum(tgt_padding_mask)

    if calculate_gradients:
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    return loss.item()

def train():
    total_loss = 0
    num_batches = len(train_dataloader)
    for batch in train_dataloader:
        loss = process_batch(batch, calculate_gradients=True)
        total_loss += loss

    return total_loss / num_batches

def test():
    total_loss = 0
    num_batches = len(test_dataloader)
    with torch.no_grad():
        for batch in test_dataloader:
            loss = process_batch(batch, calculate_gradients=False)
            total_loss += loss

    return total_loss / num_batches


def print_translation(src_sequence: Tensor, tgt_sequence: Tensor, output_sequence: Tensor):
    output_sequence = output_sequence.argmax(dim=1)
    sequences = {'Source': src_sequence, 'Target': tgt_sequence, 'Output': output_sequence}

    for domain, sequence in sequences.items():
        eos_index = (sequence == eos_idx).nonzero(as_tuple=True)
        if len(eos_index) > 0:
            first_eos_index = eos_index[0][0].item()
            sequence_before_eos_tensor = sequence[:first_eos_index]
        else:
            sequence_before_eos_tensor = sequence
            
        if domain == 'Source':   
            # Ex) "Source: A man who just came from a swim"
            print(f"{domain}: {' '.join([vocab[src_language].get_itos()[idx] for idx in sequence_before_eos_tensor])}")
        else:
            print(f"{domain}: {' '.join([vocab[tgt_language].get_itos()[idx] for idx in sequence_before_eos_tensor])}")
        
    print("\n")


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def forward(self, batch):
        embeddings = self.embedding(batch)

        return embeddings

class PositionalEncoding(nn.Module):
    def __init__(self, max_length, embedding_dim):
        super(PositionalEncoding, self).__init__()
        self.positional_encoding = self.create_positional_encoding(max_length, embedding_dim)

    def get_angles(self, max_length, embedding_dim):
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.pow(10000, torch.arange(0, embedding_dim, 2).float() / embedding_dim)

        return position / div_term

    def create_positional_encoding(self, max_length, embedding_dim):
        angles = self.get_angles(max_length, embedding_dim)
        positional_encoding = torch.zeros(max_length, embedding_dim)
        positional_encoding[:, 0::2] = torch.sin(angles)
        positional_encoding[:, 1::2] = torch.cos(angles)

        return positional_encoding

    def forward(self, x):
        return x + self.positional_encoding[:x.size(1), :].detach()

class ScaledDotProductAttention(nn.Module):
    def __init__(self, embedding_dim, attention_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.scale = torch.sqrt(torch.tensor(attention_dim).float())
        self.Wq = nn.Linear(embedding_dim, attention_dim)
        self.Wk = nn.Linear(embedding_dim, attention_dim)
        self.Wv = nn.Linear(embedding_dim, attention_dim)

    def forward(self, query, key, value, mask=None):
        query = self.Wq(query)
        key = self.Wk(key)
        value = self.Wv(value)

        key = key.transpose(1, 2)
        attention_score = torch.bmm(query, key) / self.scale
        attention_score = attention_score.masked_fill(mask == 0, -1e10)
        attention_distribution = torch.softmax(attention_score, dim=-1)
        attention_value = torch.bmm(attention_distribution, value)

        return attention_value

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadAttention, self,).__init__()
        self.attention_dim = embedding_dim // num_heads
        assert self.attention_dim * num_heads == embedding_dim

        self.attention_heads = nn.ModuleList([ScaledDotProductAttention(embedding_dim, self.attention_dim)
                                               for _ in range(num_heads)])
        self.Wo = nn.Linear(embedding_dim, embedding_dim)
        self.num_heads = num_heads

    def forward(self, query, key, value, mask=None):
        attention_values = []

        attention_values = [attention_head(query, key, value, mask) for attention_head in self.attention_heads]
        concatenated_attention_values = torch.cat(attention_values, dim=-1)
        output = self.Wo(concatenated_attention_values)

        return output
        

class ResidualConnection(nn.Module):
    def __init__(self):
        super(ResidualConnection, self).__init__()
        pass

    def forward(self, inputs):
        pass

class LayerNormalization(nn.Module):
    def __init__(self):
        super(LayerNormalization, self).__init__()
        pass

    def forward(self, inputs):
        pass

class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self):
        super(FeedForwardNeuralNetwork, self).__init__()
        pass

    def forward(self, inputs):
        pass

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        pass

    def forward(self):
        pass
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        pass

    def forward(self):
        pass

class StackedEncoder(nn.Module):
    def __init__(self):
        super(StackedEncoder, self).__init__()
        pass

    def forward(self):
        pass

class StackedDecoder(nn.Module):
    def __init__(self):
        super(StackedDecoder, self).__init__()
        pass

    def forward(self):
        pass
 

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dim = 512
    num_layers = 6
    batch_size = 64
    max_length = 20
    learning_rate = 0.001
    src_language = "de"
    tgt_language = "en"
    language_pair = (src_language, tgt_language)
    special_symbols = ['<unk>', '<sos>', '<eos>', '<pad>']
    unk_idx, sos_idx, eos_idx, pad_idx = 0, 1, 2, 3
    tokenizer = {src_language : get_tokenizer('spacy', language='de_core_news_sm'),
                 tgt_language : get_tokenizer('spacy', language='en_core_web_sm')}

    # Each is a dictionary which has key: language
    vocab, vocab_size = create_vocab_from_text_data()
    
    # Multi30k test set has encoding problem, so we use train and valid set for training and testing
    train_data, test_data = Multi30k(split=('train', 'valid'), language_pair=language_pair)        
    train_dataloader = create_dataloader_from_text_data(train_data, batch_size)
    test_dataloader = create_dataloader_from_text_data(test_data, batch_size)
    
    
    # Training
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    
    for epoch in range(100):
        train_loss = train()
        
        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(train_loss))
    
    # Test
    test_loss = test()

    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    

        
    
    
