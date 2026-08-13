from __future__ import annotations

import abc
import weakref
from typing import TYPE_CHECKING, Optional, Set, Type

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


class BatchedPenalizerOrchestrator:
    def __init__(
        self,
        vocab_size: int,
        batch: ScheduleBatch,
        penalizers: Set[Type[_BatchedPenalizer]],
    ):
        self.vocab_size = vocab_size
        self._batch_ref = weakref.ref(batch)
        self.device = batch.device
        self.penalizers = {Penalizer: Penalizer(self) for Penalizer in penalizers}

        is_required = False
        for penalizer in self.penalizers.values():
            pen_is_required = penalizer.prepare_if_required()
            is_required |= pen_is_required
        self.is_required = is_required

    @property
    def batch(self) -> ScheduleBatch | None:
        return self._batch_ref()

    @batch.setter
    def batch(self, value: Optional[ScheduleBatch]):
        if value is None:
            self._batch_ref = lambda: None
        else:
            self._batch_ref = weakref.ref(value)

    def reqs(self):
        return self.batch.reqs

    def cumulate_output_tokens(
        self, output_ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        """
        Feed the output tokens to the penalizers.

        Args:
            output_ids (torch.Tensor): ``[batch_size]`` when a step commits one
                token per request, or ``[batch_size, max_new]`` when speculative
                decoding commits several.
            mask: Optional boolean tensor shaped like ``output_ids``. It marks
                real token positions in a ragged speculative step so padding is
                a no-op for every penalizer.
        """
        if output_ids.ndim == 1:
            if mask is not None and mask.shape != output_ids.shape:
                raise ValueError(
                    "1-D penalty mask must have the same shape as output_ids, "
                    f"got {tuple(mask.shape)} and {tuple(output_ids.shape)}."
                )
            for penalizer in self.penalizers.values():
                penalizer.cumulate_output_tokens(output_ids=output_ids, mask=mask)
            return

        if output_ids.ndim != 2:
            raise ValueError(
                "output_ids must be 1-D or 2-D, "
                f"got shape={tuple(output_ids.shape)}."
            )
        if mask is not None and mask.shape != output_ids.shape:
            raise ValueError(
                "2-D penalty mask must have the same shape as output_ids, "
                f"got {tuple(mask.shape)} and {tuple(output_ids.shape)}."
            )

        # Accumulate each accepted position in order. Keeping the penalizer API
        # 1-D avoids scatter_'s duplicate-index behavior for repeated tokens in
        # the same speculative run (for example [x, x, y]).
        for step in range(output_ids.shape[1]):
            step_mask = None if mask is None else mask[:, step]
            for penalizer in self.penalizers.values():
                penalizer.cumulate_output_tokens(
                    output_ids=output_ids[:, step], mask=step_mask
                )

    def apply(self, logits: torch.Tensor, repeat: Optional[int] = None):
        """
        Apply all penalizers to the logits in-place.

        Args:
            logits: The logits tensor to apply penalties to.
            repeat: If set (speculative decoding), per-request penalties are
                expanded via repeat_interleave to match the draft token layout.
                Additive penalties are captured into a zeros tensor, expanded,
                then added; scaling penalties are accumulated, expanded, then
                applied directly.
        """
        if repeat is None:
            for penalizer in self.penalizers.values():
                penalizer.apply(logits)
        else:
            # Additive: capture into zeros, expand, add
            bs = logits.shape[0] // repeat
            additive = torch.zeros(
                (bs, logits.shape[1]), dtype=torch.float32, device=logits.device
            )
            self.accumulate_additive_penalties(additive)
            logits.add_(torch.repeat_interleave(additive, repeat, dim=0))
            # Scaling: accumulate, expand, apply
            accumulated = self.accumulate_scaling_penalties()
            if accumulated is not None:
                from sglang.srt.sampling.penaltylib.repetition_penalty import (
                    apply_scaling_penalties,
                )

                expanded = torch.repeat_interleave(accumulated, repeat, dim=0)
                apply_scaling_penalties(logits, expanded)

    def accumulate_additive_penalties(self, logits: torch.Tensor):
        """Apply only additive (non-multiplicative) penalizers."""
        for penalizer in self.penalizers.values():
            if not penalizer.is_multiplicative:
                penalizer.apply(logits)

    def accumulate_scaling_penalties(self) -> Optional[torch.Tensor]:
        """Accumulate all multiplicative penalty tensors into one, or None if none active."""
        result = None
        for penalizer in self.penalizers.values():
            if not penalizer._is_prepared or not penalizer.is_multiplicative:
                continue
            if result is None:
                result = penalizer.get_scaling_penalties().clone()
            else:
                result *= penalizer.get_scaling_penalties()
        return result

    def get_repetition_penalty_factors(self) -> Optional[torch.Tensor]:
        """Snapshot the per-request repetition factor, if repetition is active.

        This deliberately exposes the scalar separately from the accumulated
        scaling table: an all-ones table cannot distinguish a token that is
        unseen from a request whose repetition factor itself is one.
        """
        # Import locally to avoid the repetition penalizer's dependency on this
        # module during class construction.
        from sglang.srt.sampling.penaltylib.repetition_penalty import (
            BatchedRepetitionPenalizer,
        )

        penalizer = self.penalizers.get(BatchedRepetitionPenalizer)
        if penalizer is None or not penalizer.is_prepared():
            return None
        return penalizer.get_repetition_penalty_factors().clone()

    def filter(self, keep_indices: torch.Tensor):
        """
        Filter the penalizers based on the indices to keep in the batch.

        Args:
            keep_indices (torch.Tensor): Tensor of indices to keep in the batch.
        """
        if not self.is_required:
            return

        if len(keep_indices) == 0:
            # No requests left in the batch, fully release orchestrator resources
            self.release()
            return

        is_required = False
        for penalizer in self.penalizers.values():
            tmp_is_required = penalizer.is_required()
            is_required |= tmp_is_required
            if tmp_is_required:
                penalizer.filter(keep_indices=keep_indices)
            else:
                penalizer.teardown()
        self.is_required = is_required

    # Resource management helpers
    def release(self) -> None:
        """Release all penalizers and break references so GC can reclaim promptly."""
        for penalizer in self.penalizers.values():
            penalizer.teardown()
        self.penalizers.clear()
        # Break reference to ScheduleBatch
        self._batch_ref = None
        self.is_required = False

    # Context manager support
    def __enter__(self) -> BatchedPenalizerOrchestrator:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def merge(self, their: BatchedPenalizerOrchestrator):
        """
        Merge the penalizers of another orchestrator into this one.

        Note that this function **must** be called _before_ self.batch.reqs is updated (filtered).
        Each unprepared penalizers would have to be prepared (creating tensors, etc.) first before merging.
        This step requires the original batch.reqs, before it gets merged with other batch.reqs.

        Args:
            their (BatchedPenalizerOrchestrator): The orchestrator to merge into this one.
        """
        if not self.is_required and not their.is_required:
            return

        self.is_required = True
        for penalizer, their_penalizer in their.penalizers.items():
            self.penalizers[penalizer].merge(their_penalizer)


class _BatchedPenalizer(abc.ABC):
    """
    An abstract class for a batched penalizer.
    """

    is_multiplicative: bool = False

    def __init__(self, orchestrator: BatchedPenalizerOrchestrator):
        self._orchestrator_ref: weakref.ReferenceType[BatchedPenalizerOrchestrator] = (
            weakref.ref(orchestrator)
        )
        self._is_prepared = False

    @property
    def orchestrator(self) -> BatchedPenalizerOrchestrator:
        orch: Optional[BatchedPenalizerOrchestrator] = self._orchestrator_ref()
        # This should never happen, but we need to handle it gracefully
        if orch is None:
            raise RuntimeError(
                "BatchedPenalizerOrchestrator has been garbage-collected"
            )
        return orch

    def is_prepared(self) -> bool:
        return self._is_prepared

    def is_required(self) -> bool:
        return self._is_required()

    def prepare(self):
        if not self._is_prepared:
            self._prepare()
            self._is_prepared = True

    def prepare_if_required(self):
        if self._is_required():
            self.prepare()
            return True
        else:
            return False

    def teardown(self):
        self._teardown()
        self._is_prepared = False

    def cumulate_output_tokens(
        self, output_ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        if not self._is_prepared:
            return

        self._cumulate_output_tokens(output_ids=output_ids, mask=mask)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._is_prepared:
            return

        self._apply(logits=logits)

    def filter(self, keep_indices: torch.Tensor):
        if not self._is_prepared:
            return

        self._filter(keep_indices=keep_indices)

    def merge(self, their: _BatchedPenalizer):
        if not self._is_prepared and not their._is_prepared:
            return

        self.prepare()
        their.prepare()
        self._merge(their)

    @abc.abstractmethod
    def _is_required(self) -> bool:
        """
        Check if the penalizer is required to be prepared.
        """
        pass

    @abc.abstractmethod
    def _prepare(self):
        """
        Prepare the penalizer.
        Usually, this is where the penalizer initializes its tensors.
        """
        pass

    @abc.abstractmethod
    def _cumulate_output_tokens(
        self, output_ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        """
        Cumulate the output tokens.
        Orchestrator will call this function to feed the output tokens to the penalizer.

        ``output_ids`` is always ``[batch_size]``. When provided, ``mask`` is a
        ``[batch_size]`` boolean tensor; rows set to False carry padding and must
        not alter the accumulated state.
        """
        pass

    @abc.abstractmethod
    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply the penalizer to the logits.
        Penalizers can modify the logits in-place if needed.
        """
        pass

    def get_scaling_penalties(self) -> torch.Tensor:
        """
        Return the accumulated scaling penalty tensor for multiplicative penalizers.
        Only meaningful when is_multiplicative is True. Subclasses should override.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _filter(self, keep_indices: torch.Tensor):
        """
        Filter the penalizer (tensors or underlying data) based on the indices to keep in the batch.
        """
        pass

    @abc.abstractmethod
    def _merge(self, their: _BatchedPenalizer):
        """
        Merge the penalizer with another penalizer.
        """
        pass

    @abc.abstractmethod
    def _teardown(self):
        """
        Teardown the penalizer.
        """
        pass
