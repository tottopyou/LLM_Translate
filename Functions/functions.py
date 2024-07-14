import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer_en = get_tokenizer('spacy', language='en_core_web_sm')
tokenizer_uk = get_tokenizer('spacy', language='uk_core_news_sm')

def build_vocab(sentences, tokenizer):
    counter = Counter()
    for sentence in sentences:
        counter.update(tokenizer(sentence))
    counter.update(['<unk>', '<pad>', '<bos>', '<eos>'])
    return Vocab(counter)

def tokenize_and_index(sentence, vocab, tokenizer):
    return [vocab['<bos>']] + [vocab[token] for token in tokenizer(sentence)] + [vocab['<eos>']]

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        print(f"Batch {i+1} - src shape: {src.shape}, trg shape: {trg.shape}")
        print(f"src sample: {src[:,0]}")
        print(f"trg sample: {trg[:,0]}")
        try:
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()
        except IndexError as e:
            print(f"IndexError: {e}")
            print(f"src: {src}",'\n len src:', len(src))
            print(f"trg: {trg}",'\n len trg:', len(trg))
            raise

    return epoch_loss / len(iterator)

def translate_sentence(sentence, src_vocab, trg_vocab, model, device, max_len=50):
    model.eval()
    tokens = [token.lower() for token in tokenizer_en(sentence)]
    tokens = [src_vocab['<bos>']] + [src_vocab[token] for token in tokens] + [src_vocab['<eos>']]

    src_tensor = torch.LongTensor(tokens).unsqueeze(1).to(device)
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    trg_indexes = [trg_vocab['<bos>']]
    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)

        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)

        if pred_token == trg_vocab['<eos>']:
            break

    trg_tokens = [trg_vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:-1]