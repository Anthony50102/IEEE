"""End-to-end ROM model: snippet -> context -> coefficients -> rollout.

Author: Anthony Poole

Composes the five primitives. Forward pass:
    1. project snippet to reduced space        (basis.project)
    2. encode reduced snippet to context       (encoder.forward)
    3. emit coefficients                       (head.forward)
    4. integrate forward in reduced space      (integrator.rollout)
    5. (optional) lift to physical space       (basis.lift)

The framework is generic; HW-specific operator wiring is bound at
construction time via the operator factory.
"""

raise NotImplementedError("rom.model: implement Phase 3")
