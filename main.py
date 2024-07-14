import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
import torch.optim as optim
import torch.nn as nn
from Models import Decoder,Encoder,Seq2Seq
import os
from torch.nn.utils.rnn import pad_sequence
from Functions import train, translate_sentence, build_vocab, tokenize_and_index

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Active device:", device)

# Load the dataset
data = pd.read_csv('Dataset/Sentence_pairs_EN_UA.tsv', sep='\t', header=None, usecols=[1, 3], names=['english', 'ukrainian'])

# Display the first few rows
print(data.head())

# Tokenizers
tokenizer_en = get_tokenizer('spacy', language='en_core_web_sm')
tokenizer_uk = get_tokenizer('spacy', language='uk_core_news_sm')

# Build vocabulary

vocab_en = build_vocab(data['english'], tokenizer_en)
vocab_uk = build_vocab(data['ukrainian'], tokenizer_uk)

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

for batch in dataloader:
    english_batch, ukrainian_batch = batch
    english_batch = english_batch.to(device)
    ukrainian_batch = ukrainian_batch.to(device)
    print(english_batch.shape, ukrainian_batch.shape)
    break


INPUT_DIM = len(vocab_en)
OUTPUT_DIM = len(vocab_uk)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=vocab_uk['<pad>'])

epochs = 10
CLIP = 1
model_path = "trained_model.pth"
if not os.path.exists(model_path):
    print("Start the training")
    for epoch in range(epochs):
        train_loss = train(model, dataloader, optimizer, criterion, CLIP)
        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}')
    torch.save(model.state_dict(), model_path)
else:
    print("Model already trained")

sentence = "Hello, how are you?"
translated_sentence = translate_sentence(sentence, vocab_en, vocab_uk, model, device)
print(' '.join(translated_sentence))

