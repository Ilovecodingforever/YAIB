import logging
from typing import Optional

import gin
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.loader import GRUDPredictionPandasDataset, GRUDPredictionPolarsDataset
from icu_benchmarks.models.wrappers import DLPredictionWrapper


class _GRUDCell(nn.Module):
    """One GRU-D recurrent layer for batch-first sequences."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(input_dim, 3 * hidden_dim)
        self.hidden_projection = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.mask_projection = nn.Linear(input_dim, 3 * hidden_dim, bias=False)
        self.input_decay_weight = nn.Parameter(torch.zeros(input_dim))
        self.input_decay_bias = nn.Parameter(torch.zeros(input_dim))
        self.hidden_decay_weight = nn.Parameter(torch.zeros(input_dim, hidden_dim))
        self.hidden_decay_bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(
        self,
        values: Tensor,
        observed_mask: Tensor,
        elapsed_time: Tensor,
        hidden: Tensor,
        reference_means: Tensor,
        sequence_mask: Tensor,
    ) -> Tensor:
        input_decay = torch.exp(
            -F.relu(elapsed_time * self.input_decay_weight + self.input_decay_bias)
        )
        hidden_decay = torch.exp(
            -F.relu(elapsed_time @ self.hidden_decay_weight + self.hidden_decay_bias)
        )

        last_observed = reference_means.unsqueeze(0).expand(values.shape[0], -1)
        outputs = []
        for timestep in range(values.shape[1]):
            mask = observed_mask[:, timestep]
            x = values[:, timestep]
            gamma_x = input_decay[:, timestep]
            gamma_h = hidden_decay[:, timestep]
            last_observed = mask * x + (1.0 - mask) * last_observed
            x = mask * x + (1.0 - mask) * (gamma_x * last_observed + (1.0 - gamma_x) * reference_means)
            decayed_hidden = gamma_h * hidden

            gates = self.input_projection(x) + self.hidden_projection(decayed_hidden) + self.mask_projection(mask)
            update, reset, candidate = gates.chunk(3, dim=-1)
            update = torch.sigmoid(update)
            reset = torch.sigmoid(reset)
            candidate = torch.tanh(candidate + reset * decayed_hidden)
            next_hidden = (1.0 - update) * candidate + update * hidden
            hidden = torch.where(sequence_mask[:, timestep].unsqueeze(-1), next_hidden, hidden)
            outputs.append(hidden)
        return torch.stack(outputs, dim=1)


@gin.configurable
class GRUDNet(DLPredictionWrapper):
    """YAIB prediction model using GRU-D decays for dynamic variables."""

    _supported_run_modes = [RunMode.classification, RunMode.regression]
    prediction_dataset_classes = {
        "polars": GRUDPredictionPolarsDataset,
        "pandas": GRUDPredictionPandasDataset,
    }

    def __init__(
        self,
        input_size,
        hidden_dim: int,
        layer_dim: int,
        num_classes: int,
        dropout: float = 0.0,
        reference_means: Optional[Tensor] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            input_size=input_size,
            hidden_dim=hidden_dim,
            layer_dim=layer_dim,
            num_classes=num_classes,
            dropout=dropout,
            reference_means=reference_means,
            **kwargs,
        )
        self.save_hyperparameters("input_size", "hidden_dim", "layer_dim", "num_classes", "dropout")
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.dropout = nn.Dropout(dropout)
        dynamic_dim = input_size[0][-1]
        static_dim = input_size[3][-1]
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.grud = _GRUDCell(dynamic_dim, hidden_dim)
        # ponytail: only the first layer consumes missingness; later layers are ordinary GRUs.
        self.rnn_layers = nn.ModuleList(nn.GRUCell(hidden_dim, hidden_dim) for _ in range(max(layer_dim - 1, 0)))
        self.static_encoder = (
            nn.Sequential(nn.Linear(static_dim, static_dim), nn.ReLU()) if static_dim else nn.Identity()
        )
        self.logit = nn.Linear(hidden_dim + static_dim, num_classes)

        means = torch.zeros(dynamic_dim, dtype=torch.float32) if reference_means is None else torch.as_tensor(reference_means)
        if means.shape != (dynamic_dim,):
            raise ValueError(f"Expected {dynamic_dim} GRU-D reference means, got shape {means.shape}.")
        self.register_buffer("reference_means", means.float().clone())
        if reference_means is None:
            logging.warning("GRU-D was initialized without reference means; using zero means.")

    def init_hidden(self, values: Tensor) -> Tensor:
        return values.new_zeros(values.shape[0], self.hidden_dim)

    def forward(self, inputs: list[Tensor] | tuple[Tensor, ...]) -> Tensor:
        values, observed_mask, elapsed_time, static, sequence_mask = inputs
        sequence_mask = sequence_mask.bool()
        hidden = self.init_hidden(values)
        output = self.grud(values, observed_mask, elapsed_time, hidden, self.reference_means, sequence_mask)
        for layer in self.rnn_layers:
            layer_output = []
            layer_hidden = self.init_hidden(values)
            for timestep in range(output.shape[1]):
                candidate = layer(output[:, timestep], layer_hidden)
                valid = sequence_mask[:, timestep].unsqueeze(-1)
                layer_hidden = torch.where(valid, candidate, layer_hidden)
                layer_output.append(layer_hidden)
            output = torch.stack(layer_output, dim=1)
            output = self.dropout(output)

        if static.shape[-1]:
            static_context = self.static_encoder(static).unsqueeze(1).expand(-1, output.shape[1], -1)
            output = torch.cat([output, static_context], dim=-1)
        return self.logit(output)

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["grud_reference_means"] = self.reference_means.detach().cpu()

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        means = checkpoint.get("grud_reference_means")
        if means is not None:
            self.reference_means.copy_(means.to(self.reference_means.device))
