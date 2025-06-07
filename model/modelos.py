import torch.nn as nn
import torch
import torch.nn.functional as F

class TLS_LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=512, output_size=1, dropout_prob=0.2):
        super(TLS_LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out)
        last_time_step_out = lstm2_out[:, -1, :]
        out = self.fc(last_time_step_out)
        return out
    
class HybridLSTMAttentionModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=512, output_size=1, dropout_prob=0.1):
        super(HybridLSTMAttentionModel, self).__init__()
        # Capas LSTM
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        # Mecanismo de Atención
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.Tanh(),
            nn.Linear(hidden_size//2, 1),
            nn.Softmax(dim=1)
        )
        
        # Capas Fully Connected
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size//2, output_size)
        )
        
        # Inicialización de pesos
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:  # Solo para matrices
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:  # Para biases
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        # Capas LSTM
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)
        lstm2_out, _ = self.lstm2(lstm1_out)
        
        # Atención
        attention_weights = self.attention(lstm2_out)
        context_vector = torch.sum(attention_weights * lstm2_out, dim=1)
        
        # Salida final
        out = self.fc(context_vector)
        return out.squeeze(-1)  # Asegura forma (batch_size,)
      
class BidirectionalDeepLSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=512, output_size=1, dropout_prob=0.2):
        super(BidirectionalDeepLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True, num_layers=2)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),  # *2 por bidirectional
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out[:, -1, :])
        return out
        
class GRU_Model(nn.Module):
    def __init__(self, input_size=2, hidden_size=512, output_size=1, dropout_prob=0.2, num_layers=2):
        super(GRU_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Capas GRU
        self.gru1 = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.gru2 = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_prob)
        
        # Capa fully connected
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Inicialización de pesos
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        # Primera capa GRU
        gru1_out, _ = self.gru1(x)
        gru1_out = self.dropout1(gru1_out)
        
        # Segunda capa GRU
        gru2_out, _ = self.gru2(gru1_out)
        gru2_out = self.dropout2(gru2_out)
        
        # Tomar la última secuencia y pasar por FC
        out = self.fc(gru2_out[:, -1, :])
        return out

class LSTMWithSelfAttention(nn.Module):
    def __init__(self, input_dim, lstm_units, num_heads, dropout_rate=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_units, batch_first=True, bidirectional=True)
        self.self_attn = nn.MultiheadAttention(lstm_units * 2, num_heads, dropout=dropout_rate, batch_first=True)
        self.norm = nn.LayerNorm(lstm_units * 2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: (batch, seq, features)
        x, _ = self.lstm(x)
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, context):
        # query: (batch, 1, embed_dim), context: (batch, N, embed_dim)
        attn_out, _ = self.cross_attn(query, context, context)
        return self.norm(query + self.dropout(attn_out))

class ReshapeToWindows(nn.Module):
    def __init__(self, seq_len, window_size, feature_dim):
        super().__init__()
        self.seq_len = seq_len
        self.window_size = window_size
        self.feature_dim = feature_dim

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        num_windows = self.seq_len // self.window_size
        x = x[:, :num_windows * self.window_size, :]
        x = x.view(x.size(0), num_windows, self.window_size, self.feature_dim)
        return x

class ContextualLSTMTransformerFlexible(nn.Module):
    def __init__(
        self,
        seq_len,
        feature_dim,
        output_size=5,  # Para regresión
        window_size=32,
        max_neighbors=2,
        lstm_units=64,
        num_heads=4,
        embed_dim=128,
        dropout_rate=0.1
    ):
        super().__init__()
        assert window_size % 2 == 0, "window_size debe ser par"
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.output_size = output_size  # Para regresión
        self.window_size = window_size
        self.max_neighbors = max_neighbors
        self.lstm_units = lstm_units
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate

        self.reshape = ReshapeToWindows(seq_len, window_size, feature_dim)
        self.lstm_attn_block = LSTMWithSelfAttention(feature_dim, lstm_units, num_heads, dropout_rate)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout_rate)
            for _ in range(seq_len // window_size)
        ])
        self.final_dense1 = nn.Linear(embed_dim * (seq_len // window_size), 64)
        self.final_dense2 = nn.Linear(64, self.output_size)

        # Project LSTM output to embed_dim for attention
        self.project = nn.Linear(lstm_units * 2, embed_dim)

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        batch_size = x.size(0)
        num_windows = self.seq_len // self.window_size

        x = self.reshape(x)  # (batch, num_windows, window_size, feature_dim)
        x = x.reshape(-1, self.window_size, self.feature_dim)  # (batch*num_windows, window_size, feature_dim)
        x = self.lstm_attn_block(x)  # (batch*num_windows, window_size, lstm_units*2)
        x = self.project(x)  # (batch*num_windows, window_size, embed_dim)
        x = x.mean(dim=1)  # (batch*num_windows, embed_dim)
        x = x.view(batch_size, num_windows, self.embed_dim)  # (batch, num_windows, embed_dim)

        window_outputs = []
        for center_idx in range(num_windows):
            left = max(0, center_idx - self.max_neighbors)
            right = min(num_windows, center_idx + self.max_neighbors + 1)
            context_indices = [i for i in range(left, right) if i != center_idx]

            # context: (batch, len(context_indices), embed_dim)
            context = x[:, context_indices, :]
            # center: (batch, 1, embed_dim)
            center = x[:, center_idx:center_idx+1, :]

            attended = self.cross_attn_blocks[center_idx](center, context)
            window_outputs.append(attended)

        output_sequence = torch.cat(window_outputs, dim=1)  # (batch, num_windows, embed_dim)
        out = output_sequence.view(batch_size, -1)  # flatten
        out = F.relu(self.final_dense1(out))
        out = self.final_dense2(out)
        return out.squeeze(-1)