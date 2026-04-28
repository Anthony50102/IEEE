"""Unstructured U-Net operator for the DISCO-lite ablation.

Author: Anthony Poole

A small 2D U-Net that consumes a state field and emits its time derivative.
Used as a drop-in replacement for `rom.operator` so that everything
upstream and downstream (encoder, head, integrator) is identical.

The head's job in the structured case is to emit O(r^2) coefficients;
in the DISCO-lite case it must emit on the order of 10^5 - 10^6 weights for
this U-Net. That parameter-count gap is the entire empirical claim of
the paper.
"""

raise NotImplementedError("disco_lite.unet_operator: implement Phase 4")
