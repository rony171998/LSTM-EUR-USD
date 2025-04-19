import torch.nn as nn
import torch

class TLS_LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=256, output_size=1, dropout_prob=0.2):
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
    def __init__(self, input_size=2, hidden_size=256, output_size=1, dropout_prob=0.2):
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

    def forward(self, x):
        # Capas LSTM
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)
        lstm2_out, _ = self.lstm2(lstm1_out)
        
        # Atención sobre la última capa LSTM
        attention_weights = self.attention(lstm2_out)
        context_vector = torch.sum(attention_weights * lstm2_out, dim=1)
        
        # Salida final
        out = self.fc(context_vector)
        return out
    
class ResidualLSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=256, output_size=1, dropout_prob=0.2):
        super(ResidualLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)
        lstm2_out, _ = self.lstm2(lstm1_out)
        # Skip connection desde la entrada de la segunda LSTM
        out = self.fc(lstm2_out[:, -1, :] + lstm1_out[:, -1, :])  # Suma residual
        return out
    
class BidirectionalDeepLSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=256, output_size=1, dropout_prob=0.2):
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
    
class TemporalAutoencoderLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=256, output_size=1, dropout_prob=0.2):
        super(TemporalAutoencoderLSTM, self).__init__()
        # Encoder
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2)
        # Decoder (para reconstrucción)
        self.decoder = nn.LSTM(hidden_size, input_size, batch_first=True)
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, output_size)
        )
        
    def forward(self, x):
        # Encoding
        encoded, _ = self.encoder(x)
        # Decoding (opcional, se puede usar como regularización)
        decoded, _ = self.decoder(encoded)
        # Predicción
        out = self.predictor(encoded[:, -1, :])
        return out, decoded  # Devuelve ambos para pérdida compuesta
    
class GRU_Model(nn.Module):
    def __init__(self, input_size=2, hidden_size=256, output_size=1, dropout_prob=0.2, num_layers=2):
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