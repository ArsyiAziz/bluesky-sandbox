@dataclass(frozen=True)
class AutoCostConstraintTaskInfoProvider:
    """Expose constraints with dense intrinsic cost for zero-violation training.

    ``extrinsic_cost_fn`` defines the true task violation. ``intrinsic_cost_fn``
    optionally adds a nonnegative risk signal before the true violation occurs.
    The replay buffer trains from ``cost`` while diagnostics can still separate
    true violation from intrinsic risk.
    """

    names: tuple[str, ...]
    limits: np.ndarray
    extrinsic_cost_fn: ConstraintFn
    intrinsic_cost_fn: ConstraintFn | None = None
    intrinsic_weight: float = 1.0
    key: str = "constraints"
    diagnostics_key: str = "safety_cost"
    # Optional per-intruder keep predicate ``keep_fn(own_idx, other_indices) ->
    # bool mask``, sharing the cost's severity so the kept set is exactly
    # "contributes to the cost". Consumed by the observation assembler (via
    # ``keep_mask``) to attach an ``intruders_keep`` mask to each obs for the
    # actor's subset-augmentation envelope. None (or no provider) leaves the key
    # absent - the actor may then subset freely.
    keep_fn: Any = None

    def keep_mask(self, own_idx: int, other_indices: Any) -> Any:
        """Risk-envelope keep mask over ``other_indices``, or None if unset."""
        return None if self.keep_fn is None else self.keep_fn(own_idx, other_indices)

    def __post_init__(self) -> None:
        limits = np.asarray(self.limits, dtype=np.float32)
        object.__setattr__(self, "limits", limits)
        if limits.ndim != 1:
            raise ValueError(f"constraint limits must be 1-D, got shape {limits.shape}")
        if len(self.names) != limits.shape[0]:
            raise ValueError(
                "constraint names and limits must have the same length, got "
                f"{len(self.names)} names and {limits.shape[0]} limits"
            )
        if not callable(self.extrinsic_cost_fn):
            raise ValueError("extrinsic_cost_fn must be callable")
        if self.intrinsic_cost_fn is not None and not callable(self.intrinsic_cost_fn):
            raise ValueError("intrinsic_cost_fn must be callable when provided")
        if self.intrinsic_weight < 0.0:
            raise ValueError("intrinsic_weight must be non-negative")
        if not self.key:
            raise ValueError("constraint key must be non-empty")
        if not self.diagnostics_key:
            raise ValueError("diagnostics key must be non-empty")

    def __call__(
        self,
        obs: BaseObs,
        action: np.ndarray | None,
        info: BaseAgentInfo,
        context: AgentStepContext,
        rng: np.random.Generator,
    ) -> None:
        extrinsic_cost = self._cost_array(
            self.extrinsic_cost_fn(obs, action, info, context, rng),
            "extrinsic",
        )
        if self.intrinsic_cost_fn is None:
            intrinsic_cost = np.zeros_like(extrinsic_cost)
        else:
            intrinsic_cost = self._cost_array(
                self.intrinsic_cost_fn(obs, action, info, context, rng),
                "intrinsic",
            )

        if self.intrinsic_weight == 1.0:
            weighted_intrinsic = intrinsic_cost
        else:
            weighted_intrinsic = intrinsic_cost * float(self.intrinsic_weight)
        cost = np.maximum(extrinsic_cost, weighted_intrinsic).astype(
            np.float32,
            copy=False,
        )
        info["task"][self.key] = {
            "cost": cost,
            "extrinsic_cost": extrinsic_cost,
            "intrinsic_cost": intrinsic_cost,
            "names": self.names,
            "limits": self.limits,
            "violated": extrinsic_cost > self.limits,
        }
        info["task"][self.diagnostics_key] = {
            "names": self.names,
            "extrinsic_cost": extrinsic_cost,
            "intrinsic_cost": intrinsic_cost,
            "total_cost": cost,
        }

    def _cost_array(self, value: np.ndarray, label: str) -> np.ndarray:
        cost = np.asarray(value, dtype=np.float32)
        if cost.shape != self.limits.shape:
            raise ValueError(
                f"{label} cost shape must match limits shape "
                f"{self.limits.shape}, got {cost.shape}"
            )
        return np.maximum(cost, 0.0).astype(np.float32)
