"""LR encode/decode contracts. Basis and zero policy belong to fitted state.

Decode returns relative positive parts, not restored counts. Explicit Other
changes the dimension; ILR balances must not be labelled as original variants.
"""
from typing import Protocol
from .contracts import Array, FitScope, InitializedPanel


class Representation(Protocol):
    def fit(self, values: InitializedPanel, scope: FitScope) -> None: ...

    def encode(self, values: Array) -> Array: ...

    def decode_relative(self, latent: Array) -> Array: ...
