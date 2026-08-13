"""Regression tests for sampling penalties on the DSPARK/DFlash verify path."""

import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    verify_logits_adjustments_are_noop,
)
from sglang.srt.speculative.spec_utils import spec_prepare_for_decode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ForwardSamplingInfo:
    """The forward-only shape produced by SamplingBatchInfo.copy_for_forward."""

    def __init__(self, additive=None, scaling=None, repetition_factors=None):
        self.acc_additive_penalties = additive
        self.acc_scaling_penalties = scaling
        self.acc_repetition_penalty_factors = repetition_factors
        self.has_custom_logit_processor = False
        self.penalizer_orchestrator = None
        self.grammar_mask = None
        self.logit_bias = None
        tensor = next(
            value
            for value in (additive, scaling, repetition_factors)
            if value is not None
        )
        self._batch_size = tensor.shape[0]

    def __len__(self):
        return self._batch_size


class TestDSparkPenaltyAdjustments(CustomTestCase):
    def test_precomputed_additive_and_scaling_penalties_are_applied(self):
        logits = torch.tensor(
            [[4.0, -4.0, 2.0], [8.0, -8.0, -2.0]], dtype=torch.float32
        )
        sampling_info = _ForwardSamplingInfo(
            additive=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
            scaling=torch.tensor([[2.0, 4.0, 1.0]], dtype=torch.float32),
        )

        apply_dflash_verify_logits_adjustments(
            next_token_logits=logits,
            sampling_info=sampling_info,
            draft_token_num=2,
        )

        torch.testing.assert_close(
            logits,
            torch.tensor([[2.5, -8.0, 5.0], [4.5, -24.0, 1.0]], dtype=torch.float32),
        )

    def test_precomputed_penalties_disable_folded_noop_path(self):
        for field in ("additive", "scaling", "repetition_factors"):
            with self.subTest(field=field):
                kwargs = {field: torch.ones((1, 4), dtype=torch.float32)}
                sampling_info = _ForwardSamplingInfo(**kwargs)
                self.assertFalse(verify_logits_adjustments_are_noop(sampling_info))

    def test_repetition_penalty_is_causal_and_set_based_within_chain(self):
        # Chain [anchor, x, x, y]. Row 0 predicts before x is seen; rows 1+
        # see x, but the second x must not turn the factor into r^2. Row 3
        # also sees y for the first time.
        verify_ids = torch.tensor([[0, 3, 3, 4]], dtype=torch.int64)
        logits = torch.full((4, 6), 8.0, dtype=torch.float32)
        logits[3, 3] = -8.0
        logits[3, 4] = -8.0
        sampling_info = _ForwardSamplingInfo(
            scaling=torch.ones((1, 6), dtype=torch.float32),
            repetition_factors=torch.tensor([[2.0]], dtype=torch.float32),
        )

        apply_dflash_verify_logits_adjustments(
            next_token_logits=logits,
            sampling_info=sampling_info,
            draft_token_num=4,
            verify_token_ids=verify_ids,
        )

        self.assertEqual(logits[0, 3].item(), 8.0)
        self.assertEqual(logits[1, 3].item(), 4.0)
        self.assertEqual(logits[2, 3].item(), 4.0)
        self.assertEqual(logits[3, 3].item(), -16.0)
        self.assertEqual(logits[2, 4].item(), 8.0)
        self.assertEqual(logits[3, 4].item(), -16.0)

    def test_compact_invalid_suffix_does_not_observe_draft_candidates(self):
        verify_ids = torch.tensor([[0, 3, 4], [0, -1, -1]], dtype=torch.int64)
        logits = torch.full((6, 6), 8.0, dtype=torch.float32)
        sampling_info = _ForwardSamplingInfo(
            scaling=torch.ones((2, 6), dtype=torch.float32),
            repetition_factors=torch.tensor([[2.0], [2.0]], dtype=torch.float32),
        )

        apply_dflash_verify_logits_adjustments(
            next_token_logits=logits,
            sampling_info=sampling_info,
            draft_token_num=3,
            verify_token_ids=verify_ids,
            valid_lens=torch.tensor([3, 1], dtype=torch.int64),
        )

        strided = logits.view(2, 3, 6)
        self.assertEqual(strided[0, 1, 3].item(), 4.0)
        self.assertEqual(strided[0, 2, 4].item(), 4.0)
        torch.testing.assert_close(strided[1], torch.full((3, 6), 8.0))

    @patch(
        "sglang.srt.speculative.spec_utils.mamba_extra_buffer_lazy_enabled",
        return_value=False,
    )
    def test_dflash_family_prepare_accumulates_penalty_tokens(self, _mock_mamba):
        batch = SimpleNamespace(
            spec_algorithm=SimpleNamespace(is_dflash_family=lambda: True),
            spec_info=SimpleNamespace(prepare_for_decode=MagicMock()),
            sampling_info=SimpleNamespace(
                penalizer_orchestrator=SimpleNamespace(is_required=True)
            ),
            cumulate_penalty_output_tokens=MagicMock(),
        )

        spec_prepare_for_decode(batch)

        batch.cumulate_penalty_output_tokens.assert_called_once_with()
        batch.spec_info.prepare_for_decode.assert_called_once_with(batch)

    def test_overlap_pending_penalty_result_requires_predrain(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.result_queue = deque([(object(), object())])
        scheduler.last_batch = SimpleNamespace(
            spec_algorithm=SimpleNamespace(is_dflash_family=lambda: True),
            sampling_info=SimpleNamespace(
                penalizer_orchestrator=SimpleNamespace(is_required=True)
            ),
        )
        self.assertTrue(scheduler._has_pending_dflash_penalty_result())

        scheduler.last_batch.sampling_info.penalizer_orchestrator.is_required = False
        self.assertFalse(scheduler._has_pending_dflash_penalty_result())

    def test_overlap_predrain_happens_before_next_batch_prepare(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.gracefully_exit = False
        scheduler.request_receiver = MagicMock()
        scheduler.request_receiver.recv_requests.side_effect = [[], StopIteration]
        scheduler.process_input_requests = MagicMock()
        scheduler._engine_paused = False
        scheduler.result_queue = deque([(object(), object())])
        scheduler.running_batch = object()
        scheduler.last_batch = SimpleNamespace(
            spec_algorithm=SimpleNamespace(is_dflash_family=lambda: True),
            sampling_info=SimpleNamespace(
                penalizer_orchestrator=SimpleNamespace(is_required=True)
            ),
        )
        scheduler.is_generation = False
        order = []
        scheduler.process_batch_result = lambda *_: order.append("result")

        def get_next_batch_to_run(**_kwargs):
            order.append("prepare")
            return SimpleNamespace(running_batch=None, batch_to_run=None)

        scheduler.get_next_batch_to_run = get_next_batch_to_run
        scheduler.is_disable_overlap_for_batch = MagicMock(return_value=False)

        with self.assertRaises(StopIteration):
            scheduler.event_loop_overlap()

        self.assertEqual(order[:2], ["result", "prepare"])


if __name__ == "__main__":
    unittest.main()
