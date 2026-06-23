import torch
import torch.nn as nn


class LSTM_RTMIN(nn.Module):
    """
    LSTM backbone embedded into the proposed RT-MIN framework.
    The framework consists of:
    1. temporal backbone,
    2. monitoring-point interaction module,
    3. rainfall-response coupling module,
    4. adaptive fusion and prediction module.
    """
    def __init__(
        self,
        num_points=6,
        hidden_dim=64,
        node_dim=64,
        num_lstm_layers=2,
        dropout=0.2
    ):
        super().__init__()

        self.backbone = LSTMBackbone(
            input_dim=num_points * 2,
            hidden_dim=hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout
        )

        self.mpim = MonitoringPointInteractionModule(
            node_dim=node_dim,
            num_heads=4,
            dropout=dropout
        )

        self.rrcm = RainfallResponseCouplingModule(
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.afpm = AdaptiveFusionPredictionModule(
            hidden_dim=hidden_dim,
            num_points=num_points,
            dropout=dropout
        )

        self.mp_proj = nn.Linear(node_dim, hidden_dim) if node_dim != hidden_dim else nn.Identity()

    def forward(self, disp_seq, rain_seq):
        """
        disp_seq: [B, T, N, 2]
        rain_seq: [B, T, 1]
        """
        B, T, N, C = disp_seq.shape

        disp_flat = disp_seq.reshape(B, T, N * C)

        h_t = self.backbone(disp_flat)
        h_m = self.mp_proj(self.mpim(disp_seq))
        h_r = self.rrcm(h_t, h_m, rain_seq)

        pred_x, pred_y, alpha = self.afpm(h_t, h_m, h_r)

        return pred_x, pred_y, alpha