import gensim
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
random.seed(20)
torch.manual_seed(20)

BATCH_SIZE = 16 # 批次大小
EPOCHS = 10  # 迭代轮数
WINDOWS_SIZE = [2, 4]  # 滑动窗口大小
EMBEDDING_DIM = 50
FEATURE_SIZE = 128  # 特征大小
N_CLASS = 2  # 标签类数
LEARNING_RATE = 0.001
MAX_SEN_LEN = 100
device = torch.device("mps")
loss_func = nn.CrossEntropyLoss()
loss_list, accuracy_list = [], []

def build_word2id(save_to_path=None):
    word2id = {'_PAD_': 0}
    path = ['./Dataset/train.txt', './Dataset/validation.txt']

    for _path in path:
        with open(_path, encoding='utf-8') as f:
            for line in f.readlines():
                sp = line.strip().split()
                for word in sp[1:]:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)
    if save_to_path:
        with open(save_to_path, 'w', encoding='utf-8') as f:
            for w in word2id:
                f.write(w + '\t')
                f.write(str(word2id[w]))
                f.write('\n')

    return word2id

def build_word2vec(fname, word2id, save_to_path=None):
    n_words = max(word2id.values()) + 1
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    wordid_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            wordid_vecs[word2id[word]] = model[word]
        except KeyError:
            pass
    if save_to_path:
        with open(save_to_path, 'w', encoding='utf-8') as f:
            for vec in wordid_vecs:
                vec = [str(w) for w in vec]
                f.write(' '.join(vec))
                f.write('\n')
    return wordid_vecs

class TextCNN(nn.Module):
    def __init__(self, word2vec, vocab_size, embedding_dim, feature_size, windows_size, max_len, n_class):
        super(TextCNN, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.embed.weight.data.copy_(torch.from_numpy(word2vec))
        self.conv1 = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=embedding_dim, out_channels=feature_size, kernel_size=h),
                          nn.LeakyReLU(),
                          nn.MaxPool1d(kernel_size=max_len - h + 1),
                          )
            for h in windows_size]
        )
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=feature_size * len(windows_size), out_features=n_class)

    def forward(self, x):
        x = self.embed(x)  # [batch_size, seq_len, embed_dim]
        x = x.permute(0, 2, 1)
        x = [conv(x) for conv in self.conv1]
        x = torch.cat(x, 1)
        x = x.view(-1, x.size(1))  # [batch_size, feature_size * len(windows_size)]
        x = self.dropout(x)
        x = self.fc1(x)
        return x


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=hidden_size * 2, out_features=num_classes)

    def forward(self, x):
        x = self.embed(x)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        output_fw = h_n[-2, :, :]
        output_bw = h_n[-1, :, :]
        out = torch.concat([output_fw, output_bw], dim=1)
        x = self.fc(out)
        return x

def load_data(path, word2id):
    contents_list, labels_list = [], []
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            sentences = line.strip().split()
            if sentences == []:
                continue
            label = int(sentences[0])
            content = [word2id.get(sen, 0) for sen in sentences[1:]]
            content = content[:MAX_SEN_LEN]
            if len(content) < MAX_SEN_LEN:
                content += [word2id['_PAD_']] * (MAX_SEN_LEN - len(content))
            labels_list.append(label)
            contents_list.append(content)
    contents = np.array(contents_list)
    labels = np.array(labels_list)

    labels_onehot = np.array([[0, 0]] * len(labels_list))
    for idx, val in enumerate(labels):
        if val == '0':
            labels_onehot[idx][0] = 1
        else:
            labels_onehot[idx][1] = 1

    return contents, labels, labels_onehot

class MyDataSet(Dataset):
    def __init__(self, data, labels):
        self.data = torch.LongTensor(np.array(data))
        self.labels = torch.LongTensor(np.array(labels))

    def __getitem__(self, index):
        data, label = self.data[index], self.labels[index]
        return data, label

    def __len__(self):
        return len(self.data)

def get_accuracy(model, datas, labels):
    out = model(torch.LongTensor(np.array(datas)).to(device))
    predictions = torch.max(input=out, dim=1)[1]
    y_predict = predictions.to('cpu').data.numpy()
    y_true = labels
    accuracy = accuracy_score(y_true, y_predict)
    precision = precision_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict)
    f1 = f1_score(y_true, y_predict)
    print('准确率：', accuracy, '\n正确率：', precision, '\n召回率：', recall, '\nF1值：', f1)

# 训练
def train(model, train_loader, test_loader, optimizer, train_len, test_len):
    train_loss_list, train_acc_list, test_loss_list, test_acc_list = [], [], [], []
    for epoch in range(EPOCHS):
        model.train()
        train_correct, test_correct, batch_num, train_loss, test_loss = 0, 0, 0, 0, 0
        for i, (input, label) in enumerate(train_loader):
            batch_num += 1
            input, label = input.to(device), label.to(device)
            output = model(input)
            loss = loss_func(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.max(output, 1)[1].cpu().numpy()
            label = label.cpu().numpy()
            correct = (pred == label).sum()
            train_correct += correct
            train_loss += loss.item()
        train_loss_list.append(train_loss / BATCH_SIZE)

        model.eval()
        outs, labels = [], []
        with torch.no_grad():
            for ii, (input, label) in enumerate(test_loader):
                input, label = input.to(device), label.to(device)
                output = model(input)
                loss = loss_func(output, label)
                pred = torch.max(output, 1)[1].cpu().numpy()
                label = label.cpu().numpy()
                correct = (pred == label).sum()
                test_correct += correct
                test_loss += loss.item()
                outs.append(output)
                labels.append(label)
            test_loss_list.append(test_loss / BATCH_SIZE)
            train_acc = train_correct / train_len
            test_acc = test_correct / test_len
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("Epoch:", epoch + 1,
                  "train_loss:{:.5f}, test_loss:{:.5f}, train_acc:{:.2f}%, test_acc:{:.2f}%".format(train_loss / (i + 1),
                                                                                                    test_loss / (ii + 1),
                                                                                                    train_acc * 100,
                                                                                                    test_acc * 100))

def execute():
    word2id = build_word2id('./Dataset/word2id.txt')
    word2vec = build_word2vec('./Dataset/wiki_word2vec_50.bin', word2id)
    train_contents, train_labels, _ = load_data('./Dataset/train.txt', word2id)
    val_contents, val_labels, _ = load_data('./Dataset/validation.txt', word2id)
    test_contents, test_labels, _ = load_data('./Dataset/test.txt', word2id)
    contents = np.vstack([train_contents, val_contents])
    labels = np.concatenate([train_labels, val_labels])
    train_dataloader = DataLoader(MyDataSet(contents, labels), batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(MyDataSet(test_contents, test_labels), batch_size=BATCH_SIZE, shuffle=True)

    train_len = len(contents)
    test_len = len(test_contents)

    # 构建模型
    vocab_size = len(word2id)
    model = TextCNN(word2vec=word2vec, vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, windows_size=WINDOWS_SIZE,
                    max_len=MAX_SEN_LEN, feature_size=FEATURE_SIZE, n_class=N_CLASS).to(device)
    # model = BiLSTM(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, hidden_size=MAX_SEN_LEN,
    #                num_layers=2, num_classes=N_CLASS).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

    train(model, train_dataloader, test_dataloader, optimizer, train_len, test_len)
    torch.save(model.state_dict(), './bilstm.pkl')
    get_accuracy(model, test_contents, test_labels)

if __name__ == '__main__':
    execute()

