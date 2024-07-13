import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
import torch.nn as nn
import torch.optim as optim

# Load the dataset
data = pd.read_csv('Dataset/Sentence_pairs_EN_UA.tsv', sep='\t', header=None, usecols=[1, 3], names=['english', 'ukrainian'])

# Display the first few rows
print(data.head())

# Tokenizers
tokenizer_en = get_tokenizer('spacy', language='en_core_web_sm')
tokenizer_uk = get_tokenizer('spacy', language='uk_core_news_sm')

# Build vocabulary
def build_vocab(sentences, tokenizer):
    counter = Counter()
    for sentence in sentences:
        counter.update(tokenizer(sentence))
    return Vocab(counter)

vocab_en = build_vocab(data['english'], tokenizer_en)
vocab_uk = build_vocab(data['ukrainian'], tokenizer_uk)

# Example of tokenization and indexing
def tokenize_and_index(sentence, vocab, tokenizer):
    return [vocab['<bos>']] + [vocab[token] for token in tokenizer(sentence)] + [vocab['<eos>']]

# Tokenize and index all sentences
data['english'] = data['english'].apply(lambda x: tokenize_and_index(x, vocab_en, tokenizer_en))
data['ukrainian'] = data['ukrainian'].apply(lambda x: tokenize_and_index(x, vocab_uk, tokenizer_uk))

print(data.head())

# Create Dataset and DataLoader
class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data.iloc[idx, 0]), torch.tensor(self.data.iloc[idx, 1])

def collate_fn(batch):
    english_batch, ukrainian_batch = zip(*batch)
    english_batch = pad_sequence(english_batch, padding_value=vocab_en['<pad>'])
    ukrainian_batch = pad_sequence(ukrainian_batch, padding_value=vocab_uk['<pad>'])
    return english_batch, ukrainian_batch

dataset = TranslationDataset(data)
dataloader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn)
