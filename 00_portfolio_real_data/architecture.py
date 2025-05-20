from torch import nn
import torch
# torch.manual_seed(42)  


class EnhancedLinearRegression(nn.Module):
    """
    Enhanced linear regression model with batch normalization
    for improved training stability
    """
    def __init__(self, k: int, dropout_rate=0.0):
        super().__init__()
        # Feature normalization
        self.batch_norm = nn.BatchNorm1d(k)
        # Optional dropout for regularization
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        # Linear layer - maps k features to 1 output
        self.linear = nn.Linear(in_features=k, out_features=1)
        
        # Initialize weights properly
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # x: (batch_size, N, k) -> reshape for batch norm
        batch_size, N, k = x.shape
        x_reshaped = x.reshape(-1, k)  # (batch_size*N, k)
        
        # Apply batch normalization
        x_normalized = self.batch_norm(x_reshaped)
        
        # Apply dropout if enabled
        if self.dropout is not None:
            x_normalized = self.dropout(x_normalized)
            
        # Apply linear layer
        output = self.linear(x_normalized)  # (batch_size*N, 1)
        
        # Reshape back to original dimensions
        output = output.reshape(batch_size, N)  # (batch_size, N)
        
        return output
    
    

class TwoLayerMLP(nn.Module):
    """
    Two-layer MLP model with batch normalization and optional dropout
    """
    def __init__(self, k: int, hidden_dim: int=32, dropout_rate=0.0):
        super().__init__()
        # Feature normalization
        self.batch_norm = nn.BatchNorm1d(k)
        
        # First layer: k -> hidden_dim
        self.fc1 = nn.Linear(k, hidden_dim)
        
        # Second layer: hidden_dim -> 1
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        # Activation function
        self.activation = nn.ReLU()
        
        # Optional dropout for regularization
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # Initialize weights properly
        nn.init.xavier_normal(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        # x: (batch_size, N, k) -> reshape for batch norm
        batch_size, N, k = x.shape
        x_reshaped = x.reshape(-1, k) # (batch_size*N, k)
        
        # Apply batch normalization
        x_normalized = self.batch_norm(x_reshaped)
        
        # First linear layer + activation
        x_hidden = self.fc1(x_normalized)
        x_hidden = self.activation(x_hidden)
        
        # Optional dropout 
        if self.dropout is not None:
            x_hidden = self.dropout(x_hidden)
        
        # Second linear layer (output)
        output = self.fc2(x_hidden) # (batch_size*N, 1)
        
        # Reshape back to (batch_size, N)
        output = output.reshape(batch_size, N)
        
        return output
    
    
class TwoLayerLSTM(nn.Module):
    """
    更高效的 LSTM 实现，处理时间序列数据
    """
    def __init__(self, k, hidden_dim=32, lstm_hidden_dim=64, dropout_rate=0.0):
        super(TwoLayerLSTM, self).__init__()
        self.k = k
        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        
        # LSTM 层
        self.lstm = nn.LSTM(k, lstm_hidden_dim, batch_first=True)
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.BatchNorm1d(lstm_hidden_dim),
            nn.Linear(lstm_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        # x shape: (batch_size, N, lookback, k)
        batch_size, N, lookback, k = x.shape
        
        # 重塑为 LSTM 输入: (batch_size*N, lookback, k)
        x_lstm = x.reshape(-1, lookback, k)
        
        # LSTM 处理
        lstm_out, _ = self.lstm(x_lstm)
        # 使用最后一个时间步的输出
        lstm_final = lstm_out[:, -1, :]  # shape: (batch_size*N, lstm_hidden_dim)
        
        # 全连接层处理
        output = self.fc_layers(lstm_final)  # shape: (batch_size*N, 1)
        
        # 重塑为最终输出
        output = output.view(batch_size, N)  # shape: (batch_size, N)
        
        return output
