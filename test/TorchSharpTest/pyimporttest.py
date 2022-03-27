import src.Python.importsd as importsd
import torch.nn as nn


NUM_WORDS = 100
EMBEDDING_VEC_LEN = 100
HIDDEN_SIZE = 128


class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(NUM_WORDS, EMBEDDING_VEC_LEN)
        self.lstm = nn.LSTM(EMBEDDING_VEC_LEN, HIDDEN_SIZE, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(HIDDEN_SIZE, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_embed = self.embedding(x)
        x_lstm, _ = self.lstm(x_embed)
        x_lstm_last_seq = x_lstm[:, -1, :]
        x_lstm_last_seq = self.dropout(x_lstm_last_seq)
        logits = self.dense(x_lstm_last_seq)
        out = self.sigmoid(logits)
        return out


if __name__ == '__main__':
    mylstm = LSTMModel()
    with open("lstm.dat", "rb") as f:
        sd = importsd.load_state_dict(f)
    mylstm.load_state_dict(sd)
