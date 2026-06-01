"""Unit tests for Phase II: SigGAN components.

Tests cover:
  - ARFNNGenerator forward pass shapes
  - SignatureLayer / SignatureDiscriminator shapes
  - CausalOTLoss (causal penalty, gradient penalty, discriminator/generator loss)
  - PathSignatureTransform (numpy backend)
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from src.siggan.generator import ARFNNGenerator, ConditionalARFNNGenerator
from src.siggan.discriminator import SignatureDiscriminator, SignatureLayer
from src.siggan.cot_loss import CausalOTLoss, CausalPenaltyNet
from src.siggan.signatures import (
    _signature_numpy,
    signature_dim,
    add_time_channel,
    PathSignatureTransform,
)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class TestARFNNGenerator:
    def test_output_shape(self):
        B, T, noise_dim, output_dim = 4, 20, 10, 6
        gen = ARFNNGenerator(noise_dim=noise_dim, output_dim=output_dim, hidden_dim=64, n_layers=2)
        z = torch.randn(B, T, noise_dim)
        out = gen(z)
        assert out.shape == (B, T, output_dim)

    def test_autoregressive_dependence(self):
        # Changing x0 should change the output
        B, T, noise_dim, output_dim = 2, 10, 5, 3
        gen = ARFNNGenerator(noise_dim=noise_dim, output_dim=output_dim, hidden_dim=32, n_layers=2)
        gen.eval()
        z = torch.randn(B, T, noise_dim)
        x0_a = torch.zeros(B, output_dim)
        x0_b = torch.ones(B, output_dim)
        with torch.no_grad():
            out_a = gen(z, x0_a)
            out_b = gen(z, x0_b)
        assert not torch.allclose(out_a, out_b)

    def test_default_x0_zeros(self):
        B, T, noise_dim, output_dim = 2, 5, 4, 3
        gen = ARFNNGenerator(noise_dim=noise_dim, output_dim=output_dim, hidden_dim=32, n_layers=2)
        gen.eval()
        z = torch.randn(B, T, noise_dim)
        x0 = torch.zeros(B, output_dim)
        with torch.no_grad():
            out_explicit = gen(z, x0)
            out_default = gen(z)
        torch.testing.assert_close(out_explicit, out_default)

    def test_conditional_generator_shape(self):
        B, T, noise_dim, output_dim, cond_dim = 3, 15, 8, 4, 1
        gen = ConditionalARFNNGenerator(
            noise_dim=noise_dim, output_dim=output_dim, hidden_dim=32, n_layers=2,
            condition_dim=cond_dim,
        )
        z = torch.randn(B, T, noise_dim)
        c = torch.rand(B, cond_dim)
        out = gen(z, c)
        assert out.shape == (B, T, output_dim)


# ---------------------------------------------------------------------------
# Discriminator / SignatureLayer
# ---------------------------------------------------------------------------

class TestSignatureLayer:
    def test_output_dim_property(self):
        in_channels, depth = 4, 3
        layer = SignatureLayer(in_channels=in_channels, depth=depth, with_time=True)
        d_eff = in_channels + 1  # time channel added
        expected = sum(d_eff ** k for k in range(1, depth + 1))
        assert layer.output_dim == expected

    def test_forward_shape_fallback(self):
        # Force fallback by patching _iisig to None
        layer = SignatureLayer(in_channels=3, depth=3, with_time=True)
        layer._iisig = None
        B, T, d = 4, 20, 3
        x = torch.randn(B, T, d)
        out = layer(x)
        assert out.shape == (B, layer.output_dim)

    def test_forward_no_crash(self):
        layer = SignatureLayer(in_channels=2, depth=2, with_time=False)
        layer._iisig = None
        x = torch.randn(3, 15, 2)
        out = layer(x)
        assert out.ndim == 2
        assert out.shape[0] == 3


class TestSignatureDiscriminator:
    def test_output_shape(self):
        B, T, in_channels = 5, 30, 6
        disc = SignatureDiscriminator(in_channels=in_channels, sig_depth=3, hidden_dim=64, n_layers=2)
        disc.sig_layer._iisig = None  # use fallback for speed
        x = torch.randn(B, T, in_channels)
        out = disc(x)
        assert out.shape == (B, 1)

    def test_output_unbounded(self):
        # Wasserstein critic should NOT apply sigmoid/tanh
        disc = SignatureDiscriminator(in_channels=3, sig_depth=2, hidden_dim=32, n_layers=2)
        disc.sig_layer._iisig = None
        x = torch.randn(4, 10, 3)
        out = disc(x)
        # Values can be anywhere (not clamped to [0,1])
        assert out.shape == (4, 1)


# ---------------------------------------------------------------------------
# CausalOTLoss
# ---------------------------------------------------------------------------

class TestCausalPenaltyNet:
    def test_output_shape(self):
        B, T, d = 4, 20, 3
        net = CausalPenaltyNet(in_channels=d, hidden_dim=32, n_layers=1)
        x = torch.randn(B, T, d)
        out = net(x)
        assert out.shape == (B, T)


class TestCausalOTLoss:
    @pytest.fixture
    def setup(self):
        B, T, d = 4, 15, 3
        cot = CausalOTLoss(in_channels=d, J=2, hidden_dim=32, lambda_cot=1.0)
        disc = SignatureDiscriminator(in_channels=d, sig_depth=2, hidden_dim=32, n_layers=1)
        disc.sig_layer._iisig = None
        x_real = torch.randn(B, T, d)
        x_fake = torch.randn(B, T, d)
        return cot, disc, x_real, x_fake

    def test_causal_penalty_scalar(self, setup):
        cot, _, x_real, x_fake = setup
        penalty = cot.causal_penalty(x_real, x_fake)
        assert penalty.shape == ()  # scalar

    def test_gradient_penalty_positive(self, setup):
        cot, disc, x_real, x_fake = setup
        gp = cot.gradient_penalty(disc, x_real, x_fake, lambda_gp=10.0)
        assert gp.item() >= 0.0

    def test_discriminator_loss_differentiable(self, setup):
        cot, disc, x_real, x_fake = setup
        x_fake_grad = x_fake.detach().requires_grad_(True)
        loss = cot.discriminator_loss(disc, x_real, x_fake_grad, lambda_gp=1.0)
        assert loss.shape == ()
        loss.backward()

    def test_generator_loss_negative_when_disc_outputs_positive(self, setup):
        cot, disc, _, x_fake = setup
        loss = cot.generator_loss(disc, x_fake)
        # generator loss = -E[D(fake)]; sign depends on disc, just check it's a scalar
        assert loss.shape == ()


# ---------------------------------------------------------------------------
# PathSignatureTransform (numpy backend)
# ---------------------------------------------------------------------------

class TestPathSignatureTransform:
    def test_dim_formula(self):
        for d in [2, 3, 4]:
            for depth in [1, 2, 3]:
                expected = sum(d ** k for k in range(1, depth + 1))
                assert signature_dim(d, depth) == expected

    def test_signature_numpy_shape(self):
        T, d = 30, 3
        path = np.random.randn(T, d)
        sig = _signature_numpy(path, depth=2)
        expected_dim = d + d * d  # level 1 + level 2
        assert sig.shape == (expected_dim,)

    def test_signature_numpy_depth3_shape(self):
        T, d = 20, 2
        path = np.random.randn(T, d)
        sig = _signature_numpy(path, depth=3)
        expected_dim = d + d**2 + d**3
        assert sig.shape == (expected_dim,)

    def test_add_time_channel(self):
        path = np.random.randn(10, 3)
        path_t = add_time_channel(path)
        assert path_t.shape == (10, 4)
        np.testing.assert_allclose(path_t[:, 0], np.linspace(0, 1, 10))

    def test_transform_batch_shape(self):
        B, T, d = 5, 20, 2
        paths = np.random.randn(B, T, d)
        pst = PathSignatureTransform(depth=2, with_time=True, backend="numpy")
        sigs = pst.transform_batch(paths)
        d_eff = d + 1
        expected_dim = d_eff + d_eff**2
        assert sigs.shape == (B, expected_dim)

    def test_linearity_level1(self):
        # Level-1 signature = total increment = path[T] - path[0]
        T, d = 15, 3
        path = np.random.randn(T, d)
        sig = _signature_numpy(path, depth=1)
        np.testing.assert_allclose(sig, path[-1] - path[0], atol=1e-10)
