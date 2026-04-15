import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, TabularDataset, BucketIterator
import spacy
import numpy as np

# ==========================================
# 1. 全局环境与设备设置 (修复点)
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 提前加载 spacy 模型，用于后续 predict_sentiment 函数
nlp = spacy.load('en_core_web_sm') 

# ==========================================
# 2. 数据定义与加载
# ==========================================
# 定义字段处理
TEXT = Field(tokenize='spacy', 
            tokenizer_language='en_core_web_sm',
            include_lengths=True)
# 修复点：BCEWithLogitsLoss 需要目标标签为 float 类型
LABEL = Field(sequential=False, use_vocab=False, dtype=torch.float)

# 加载数据集 (请确保 ./data/train.csv 和 test.csv 存在)
train_data, test_data = TabularDataset.splits(
    path='./data',
    train='train.csv',
    test='test.csv',
    format='csv',
    fields=[('text', TEXT), ('label', LABEL)]
)

# 构建词汇表，并下载预训练词向量
TEXT.build_vocab(train_data, 
                max_size=25000,
                vectors="glove.6B.100d")

# 修复点：生成数据迭代器 (BucketIterator)
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data), 
    batch_size=64,
    sort_within_batch=True, # 配合 pack_padded_sequence 使用，按长度降序排列批次
    sort_key=lambda x: len(x.text),
    device=device
)

# ==========================================
# 3. 模型定义
# ==========================================
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers,
                           bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        # 修复点：加入 enforce_sorted=False 作为双重保险
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.to('cpu'), enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)
    
# 模型参数配置
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2

# 初始化模型并部署到设备
model = SentimentLSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS)
model = model.to(device)

# ==========================================
# 4. 加载预训练词向量 (修复点)
# ==========================================
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

# 将未知词 <unk> 和填充词 <pad> 的权重初始化为 0
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

# ==========================================
# 5. 训练与评估配置
# ==========================================
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
criterion = criterion.to(device) # 修复点：损失函数也要放进 GPU/CPU

def accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for batch in iterator:
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += accuracy(predictions, batch.label)
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.label)
            
            epoch_loss += loss.item()
            epoch_acc += accuracy(predictions, batch.label)
            
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# ==========================================
# 6. 推理预测
# ==========================================
def predict_sentiment(model, sentence):
    model.eval() # 确保模型处于评估模式
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    
    with torch.no_grad(): # 预测时不需要计算梯度
        prediction = torch.sigmoid(model(tensor, length_tensor))
    return prediction.item()

# --- 模拟运行测试 (请在完成训练后执行这部分以获得有意义的结果) ---
"""
N_EPOCHS = 5
for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, test_iterator, criterion)
    print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
"""

positive_review = "This movie was fantastic! I really enjoyed it."
negative_review = "The film was terrible and boring."

print(f"Positive review score: {predict_sentiment(model, positive_review):.4f}")
print(f"Negative review score: {predict_sentiment(model, negative_review):.4f}")