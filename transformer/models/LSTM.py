from torch import nn


class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()

        # utilize the LSTM model in torch.nn
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.linear1 = nn.Linear(hidden_size, output_size)  # 全连接层

    def forward(self, _x, device):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.linear1(x).to(device)
        x = x.view(s, b, -1)
        return x
