# model_wrapper.py
import torch
import numpy as np
from sklearn.preprocessing import RobustScaler
from modelos import GRU_Model  # ajusta al nombre exacto de tu clase

SEQ_LENGTH = 120  # ajustar si tu modelo usa otro

class LiveModel:
    def __init__(self, model_path=None, device='cpu'):
        self.device = torch.device(device)
        if model_path:
            ckpt = torch.load(model_path, map_location=self.device)
            model_class = ckpt.get('model_class', 'GRU_Model')
            optuna = ckpt.get('optuna_params', {})
            seq_len = ckpt.get('seq_length', SEQ_LENGTH)
            self.seq_len = seq_len

            # reproducir la creación del modelo; aquí un ejemplo para GRU
            self.model = GRU_Model(
                input_size=optuna.get('input_size', 1),
                hidden_size=optuna.get('hidden_size', 64),
                output_size=1,
                dropout_prob=optuna.get('dropout_prob', 0.1)
            ).to(self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.model.eval()
            self.scaler = RobustScaler()  # si guardaste scaler, cárgalo aquí
        else:
            # modo fake (para pruebas)
            self.model = None
            self.seq_len = SEQ_LENGTH
            self.scaler = RobustScaler()

    def predict_live(self, sequence):
        """
        sequence: lista o np.array de precios (últimas seq_len)
        devuelve: (signal, stop_pips, tp_pips, confidence)
        """
        seq = np.array(sequence[-self.seq_len:], dtype=float)
        if len(seq) < self.seq_len:
            return 0, None, None, 0.0

        if self.model is None:
            # fake heuristic: si último > media última 10 -> buy
            ma10 = seq[-10:].mean()
            last = seq[-1]
            if last > ma10 * 1.0003:
                return 1, 20, 40, 0.6
            elif last < ma10 * 0.9997:
                return -1, 20, 40, 0.55
            else:
                return 0, None, None, 0.1

        # real model path:
        X = seq.reshape(-1, 1)
        Xs = self.scaler.transform(X)  # si no ajustaste scaler, mejor entrenarlo offline
        xt = torch.FloatTensor(Xs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_scaled = self.model(xt).cpu().numpy().flatten()[0]
        # desescalar si guardaste target_scaler; si no, usar pred_scaled directo
        pred_value = pred_scaled
        last = seq[-1]
        # regla simple: threshold
        if pred_value > last * 1.001:
            return 1, 20, 40, 0.6
        elif pred_value < last * 0.999:
            return -1, 20, 40, 0.55
        else:
            return 0, None, None, 0.1
