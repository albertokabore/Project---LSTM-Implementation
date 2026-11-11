import numpy as np
import pytest
import torch

from cnn_lstm.data import Normalizer, Split
from cnn_lstm.metrics import compute_metrics
from cnn_lstm.models import CNNLSTM, build_model, count_parameters, describe
from cnn_lstm.train import TrainConfig, fit, predict


@pytest.mark.parametrize(
    "arch,kwargs",
    [
        ("pixel_lstm", dict(hidden=16, layers=2)),
        ("cnn", dict(channels=(4, 8, 8))),
        ("cnn_lstm", dict(channels=(4, 8, 8), hidden=16, bidirectional=False)),
        ("cnn_lstm", dict(channels=(4, 8, 8), hidden=16, bidirectional=True, layers=2)),
    ],
)
def test_models_output_logits_per_class(arch, kwargs):
    model = build_model(arch, num_classes=8, image_size=32, **kwargs)
    out = model(torch.randn(3, 3, 32, 32))
    assert out.shape == (3, 8)


def test_cnn_lstm_sequence_is_feature_map_rows():
    model = CNNLSTM(num_classes=8, image_size=64, channels=(4, 8, 16), hidden=8)
    # 64 / 2**3 = 8 columns x 16 channels per time step
    assert model.lstm.input_size == 8 * 16
    assert model.lstm.bidirectional


def test_unknown_architecture_rejected():
    with pytest.raises(ValueError):
        build_model("transformer", 8)


def test_describe_reports_total_parameters():
    model = CNNLSTM(num_classes=8, image_size=32, channels=(4, 8, 8), hidden=8)
    text = describe(model, image_size=32)
    assert f"{count_parameters(model):,}" in text
    assert "LSTM" in text


def test_fit_learns_a_trivially_separable_problem():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 40, (64, 32, 32, 3), dtype=np.uint8)
    y = np.repeat([0, 1], 32)
    x[y == 1] += 200  # class 1 is bright
    split = Split(x, y)
    norm = Normalizer(np.full(3, 0.5, np.float32), np.full(3, 0.4, np.float32))
    model = CNNLSTM(num_classes=2, image_size=32, channels=(4, 8, 8), hidden=8, dropout=0.0)
    hist = fit(model, split, split, norm, TrainConfig(epochs=8, batch_size=16, lr=3e-3, augment=False), log=lambda _: None)
    assert hist.val_acc[-1] >= 0.95
    probs = predict(model, x, norm)
    np.testing.assert_allclose(probs.sum(1), 1.0, rtol=1e-5)


def test_compute_metrics_on_known_predictions():
    y = np.array([0, 0, 1, 1])
    probs = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.3, 0.7]])
    m = compute_metrics(y, probs, ["a", "b"])
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["per_class"]["a"]["precision"] == pytest.approx(1.0)
    assert m["per_class"]["a"]["recall"] == pytest.approx(0.5)
    assert m["per_class"]["b"]["precision"] == pytest.approx(2 / 3)
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]
    assert m["top2_accuracy"] == pytest.approx(1.0)
