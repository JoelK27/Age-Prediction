"""
Metriken für die Altersschätzung.

Enthält:
- mae: Mean Absolute Error zwischen Vorhersage und Ziel.
- mse: Mean Squared Error zwischen Vorhersage und Ziel.

Erwartung:
- p und t sind torch.Tensor derselben Form (z. B. [B] für Regression).
- Rückgabe ist ein skalarer Tensor (0-dim), der über alle Elemente mittelt.
"""

import torch
from torch import Tensor

__all__ = ["mae", "mse"]

def mae(p: Tensor, t: Tensor) -> Tensor:
    """
    Mean Absolute Error (MAE).
    p und t müssen broadcast-kompatibel sein; typischerweise gleiche Form.
    """
    # Betrag des Fehlers elementweise, dann Mittelwert über alle Elemente
    return torch.mean(torch.abs(p - t))

def mse(p: Tensor, t: Tensor) -> Tensor:
    """
    Mean Squared Error (MSE).
    p und t müssen broadcast-kompatibel sein; typischerweise gleiche Form.
    """
    # Quadratischer Fehler elementweise, dann Mittelwert über alle Elemente
    return torch.mean((p - t) ** 2)