# Copyright 2024 Quasar AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1. Configuration Class ---
class GridLNNConfig:
    """
    Configuration class for the Grid Liquid Neural Network (LNN) model.
    """
    def __init__(
        self,
        grid_size=(30, 30),
        input_channels=1, # Typically 1 for ARC grids
        num_colors=10, # 0-9
        embedding_dim=64,
        hidden_dim=64,
        evolution_steps=16, # Number of LNN evolution steps (T)
        dt=0.1, # Time step for Euler integration
        tau_floor=0.1, # Stability for time constants
        derivative_clamp_value=10.0, # Stability for state derivatives
        **kwargs
    ):
        self.grid_size = grid_size
        self.input_channels = input_channels
        self.num_colors = num_colors
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.evolution_steps = evolution_steps
        self.dt = dt
        self.tau_floor = tau_floor
        self.derivative_clamp_value = derivative_clamp_value

# --- 2. Core Modules ---

class GridInteraction(nn.Module):
    """Grid Interaction Module using a simple convolution to aggregate neighbor info."""
    def __init__(self, config: GridLNNConfig):
        super().__init__()
        # 3x3 convolution to get info from 8 neighbors + self
        self.conv = nn.Conv2d(
            in_channels=config.hidden_dim,
            out_channels=config.hidden_dim,
            kernel_size=3,
            padding=1, # Keep grid size constant
            bias=False
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Input h: (B, H, W, D) -> (B, D, H, W) for conv
        h_permuted = h.permute(0, 3, 1, 2)
        neighbor_info = self.conv(h_permuted)
        # Output: (B, D, H, W) -> (B, H, W, D)
        return neighbor_info.permute(0, 2, 3, 1)

class GridLNNCell(nn.Module):
    """A single Grid LNN cell with continuous-time dynamics and neighbor interaction."""
    def __init__(self, config: GridLNNConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim

        # Core LNN parameters: W for hidden, U for input, N for neighbors
        self.W = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.U = nn.Linear(config.embedding_dim, config.hidden_dim, bias=False)
        self.N = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False) # Neighbor influence
        self.b = nn.Parameter(torch.empty(config.hidden_dim))

        # Input-Dependent Time Constant (tau) dynamics
        self.tau_w_h = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.tau_w_u = nn.Linear(config.embedding_dim, config.hidden_dim, bias=False)
        self.tau_b = nn.Parameter(torch.empty(config.hidden_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.U.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.N.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b)
        self.tau_b.data.uniform_(-2, 2)

    def forward(self, h, u, neighbor_info):
        """
        Core ODE dynamics calculation for a single discrete step.
        h: current hidden state (B, H, W, D_h)
        u: original embedded input (B, H, W, D_e)
        neighbor_info: aggregated info from neighbors (B, H, W, D_h)
        """
        # 1. Compute Input-Dependent Time Constant (tau)
        tau_control = self.tau_w_h(h) + self.tau_w_u(u) + self.tau_b
        tau_inv = 1.0 / (self.config.tau_floor + F.softplus(tau_control))

        # 2. Compute State Update (dx/dt)
        recurrent_term = self.W(torch.tanh(h))
        input_term = self.U(u)
        neighbor_term = self.N(neighbor_info)

        # Core ODE dynamics
        dx_dt = tau_inv * (recurrent_term + input_term + neighbor_term + self.b - h)

        # 3. Apply clamping for stability
        dx_dt = torch.clamp(dx_dt, -self.config.derivative_clamp_value, self.config.derivative_clamp_value)
        return dx_dt

# --- 3. Full Grid LNN Model ---
class GridLNN(nn.Module):
    """The Grid Liquid Neural Network Model for ARC tasks."""
    def __init__(self, config: GridLNNConfig):
        super().__init__()
        self.config = config

        # Input Encoder: maps color values (0-9) to dense vectors
        self.embedding = nn.Embedding(config.num_colors, config.embedding_dim)

        # Grid Interaction Module
        self.interaction = GridInteraction(config)

        # Core LNN Cell
        self.cell = GridLNNCell(config)

        # Readout Layer: maps final hidden state to color predictions
        self.readout = nn.Linear(config.hidden_dim, config.num_colors)

    def forward_from_em(self, u: torch.Tensor):
        """
        Evolves the grid state over T steps starting from an embedding.
        u: (B, H, W, D_e) tensor of embedded inputs.
        """
        # 1. Initialize hidden state
        # h -> (B, H, W, D_h)
        h = torch.tanh(self.cell.U(u))

        # 2. Evolution Loop
        for _ in range(self.config.evolution_steps):
            # a. Get neighbor information
            neighbor_info = self.interaction(h)

            # b. Calculate state change
            dx_dt = self.cell(h, u, neighbor_info)

            # c. Update hidden state using Euler integration
            h = h + self.config.dt * dx_dt

        # 3. Decode final state to get predictions
        # logits -> (B, H, W, num_colors)
        logits = self.readout(h)

        return logits

    def forward(self, grid_input: torch.Tensor):
        """
        Evolves the grid state over T steps.
        grid_input: (B, H, W) tensor of integer color values.
        """
        # Encode input grid and run full evolution
        u = self.embedding(grid_input)
        return self.forward_from_em(u)
