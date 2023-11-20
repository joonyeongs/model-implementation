# Description: Seq2seq Model in PyTorch


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


def train():
    total_loss = 0
    num_batches = len(train_dataloader)
    for batch in train_dataloader:
        src_batch, tgt_batch = batch
        src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
        src_padding_mask, tgt_padding_mask = create_padding_mask(src_batch, tgt_batch)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        enc_hidden_states, enc_last_hidden_state = encoder(src_batch)                    # hidden_states: [batch_size, max_length, hidden_dim]
        output = decoder(src_padding_mask, enc_hidden_states, enc_last_hidden_state, tgt_batch)

        output = output[:, :-1].reshape(-1, vocab_size[tgt_language])   #[batch_size * (max_length - 1), vocab_size]
        target = tgt_batch[:, 1:].reshape(-1)                           #[batch_size * (max_length - 1)]
        tgt_padding_mask = tgt_padding_mask[:, 1:].reshape(-1)          #[batch_size * (max_length - 1)]
        train_loss = criterion(output, target)
        train_loss = torch.sum(train_loss * tgt_padding_mask) / torch.sum(tgt_padding_mask)
        train_loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += train_loss.item()
    
    return total_loss / num_batches

def test():
    num_batches = len(test_dataloader)
    total_loss = 0

    for batch in test_dataloader:
        src_batch, tgt_batch = batch
        src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
        src_padding_mask, tgt_padding_mask = create_padding_mask(src_batch, tgt_batch)

        enc_hidden_states, enc_last_hidden_state = encoder(src_batch)                    # hidden_states: [batch_size, max_length, hidden_dim]
        output = decoder(src_padding_mask, enc_hidden_states, enc_last_hidden_state, tgt_batch)

        random_integer = random.randint(0, batch_size - 1)
        print_translation(src_batch[random_integer], tgt_batch[random_integer], output[random_integer])

        output = output[:, :-1].reshape(-1, vocab_size[tgt_language])   #[batch_size * (max_length - 1), vocab_size]
        target = tgt_batch[:, 1:].reshape(-1)                           #[batch_size * (max_length - 1)]
        tgt_padding_mask = tgt_padding_mask[:, 1:].reshape(-1)          #[batch_size * (max_length - 1)]
        test_loss = criterion(output, target)
        test_loss = torch.sum(test_loss * tgt_padding_mask) / torch.sum(tgt_padding_mask)
        total_loss += test_loss.item()

    total_loss /= num_batches

    return total_loss

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

    

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size[src_language], input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        hidden_states, (last_hidden_state, _) = self.lstm(x)   # hidden_states: [batch_size, max_length, hidden_dim]

        return hidden_states, last_hidden_state                 # last_hidden_state: [num_layers, batch_size, hidden_dim]
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size[tgt_language], input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = Attention()
        self.Wc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.Wh = nn.Linear(hidden_dim, vocab_size[tgt_language])
        self.tanh = nn.Tanh()

    def forward(self, src_padding_mask, enc_hidden_states, enc_last_hidden_state, tgt_batch=None):
        dec_input = torch.empty(batch_size, 1, dtype=torch.long).fill_(sos_idx).to(device)
        dec_outputs = torch.zeros(max_length, batch_size, vocab_size[tgt_language]).to(device)
        hidden = enc_last_hidden_state
        cell = torch.zeros_like(hidden)

        for t in range(max_length):
            dec_input = self.embedding(dec_input)
            dec_output, (hidden, cell) = self.lstm(dec_input, (hidden, cell)) # decoder_output: [batch_size, 1, hidden_dim]
            context_vector = self.attention(dec_output, enc_hidden_states, enc_hidden_states, src_padding_mask)
            concatenated_vector = torch.cat((context_vector, dec_output), dim=2)  # concatenated_vector: [batch_size, 1, hidden_dim * 2]
            dec_output = self.tanh(self.Wc(concatenated_vector.squeeze(1)))
            dec_output = self.Wh(dec_output)
            dec_outputs[t] = dec_output

            if tgt_batch is not None:
                dec_input = tgt_batch[:, t].unsqueeze(1)
            else:
                dec_input = dec_output.argmax(dim=1).unsqueeze(1)

        dec_outputs = dec_outputs.permute(1, 0, 2)       # reshape to [batch_size, max_length, vocab_size]

        return dec_outputs

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_dim, hidden_dim)
        self.Ua = nn.Linear(hidden_dim, hidden_dim)
        self.va = nn.Linear(hidden_dim, 1)

    def forward(self, query, key, value, mask):
        """
        a(s, h) = va * tanh(Wa * s + Ua * h)   s: dec_hidden_state, h: hidden_states
        """
        query = query.squeeze(1)          # query: [batch_size, hidden_dim]
        query = self.Wa(query)    
        query = query.unsqueeze(1)        # query: [batch_size, 1, hidden_dim]

        key = key.reshape(-1, hidden_dim)   # key: [batch_size * max_length, hidden_dim]
        key = self.Ua(key)          
        key = key.reshape(batch_size, max_length, hidden_dim)

        sum_result = query + key           # sum_result: [batch_size, max_length, hidden_dim]
        sum_result = sum_result.reshape(-1, hidden_dim)
        attention_score = self.va(torch.tanh(sum_result))       # attention_score: [batch_size * max_length, 1]
        attention_score = attention_score.reshape(batch_size, max_length)            
        attention_score = attention_score.masked_fill(mask == 0, -1e10)
        attention_distribution = nn.Softmax(dim=1)(attention_score)  
        attention_distribution = attention_distribution.unsqueeze(1)
        context_vector = torch.bmm(attention_distribution, value)   # context_vector: [batch_size, 1, hidden_dim]
        return context_vector

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 300
    hidden_dim = 200
    num_layers = 1
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
    
    with torch.no_grad():
        test_loss = test()

    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    

        
    
    
