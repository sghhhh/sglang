import unittest
from unittest import mock

import torch
from sglang.kernels.ops.attention.dsa import tilelang_kernel as tilelang_dsa
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


class TestDSATilelangLaunchSelector(CustomTestCase):
    def test_low_smem_device_uses_v1_config(self):
        q, kv, indices = self._inputs()
        compiled_kernel = mock.Mock(return_value=torch.empty(1))

        with (
            mock.patch.object(
                tilelang_dsa,
                "_get_cuda_shared_memory_per_block_optin",
                return_value=99 * 1024,
            ),
            mock.patch.object(
                tilelang_dsa,
                "sparse_attention_fwd_kernel_v1",
                return_value=compiled_kernel,
            ) as v1,
            mock.patch.object(tilelang_dsa, "sparse_attention_fwd_kernel_v2") as v2,
        ):
            tilelang_dsa.tilelang_sparse_fwd(
                q, kv, indices, sm_scale=0.125, d_v=256
            )

        v1.assert_called_once_with(
            64,
            256,
            0,
            2048,
            sm_scale=0.125,
            block_I=64,
            num_stages=1,
            threads=128,
        )
        v2.assert_not_called()

    def test_high_smem_device_keeps_v2_config(self):
        q, kv, indices = self._inputs()
        compiled_kernel = mock.Mock(return_value=torch.empty(1))

        with (
            mock.patch.object(
                tilelang_dsa,
                "_get_cuda_shared_memory_per_block_optin",
                return_value=128 * 1024,
            ),
            mock.patch.object(tilelang_dsa, "sparse_attention_fwd_kernel_v1") as v1,
            mock.patch.object(
                tilelang_dsa,
                "sparse_attention_fwd_kernel_v2",
                return_value=compiled_kernel,
            ) as v2,
        ):
            tilelang_dsa.tilelang_sparse_fwd(
                q, kv, indices, sm_scale=0.125, d_v=256
            )

        v1.assert_not_called()
        v2.assert_called_once_with(64, 256, 0, 2048, sm_scale=0.125)

    @staticmethod
    def _inputs():
        q = torch.empty((1, 64, 256), dtype=torch.bfloat16)
        kv = torch.empty((1, 1, 256), dtype=torch.bfloat16)
        indices = torch.zeros((1, 1, 2048), dtype=torch.int32)
        return q, kv, indices


if __name__ == "__main__":
    unittest.main()
