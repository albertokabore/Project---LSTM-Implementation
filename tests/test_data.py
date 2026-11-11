import numpy as np
import pytest
import torch
from PIL import Image

from cnn_lstm.data import Augmenter, Normalizer, channel_stats, index_dataset, load_image, stratified_split


@pytest.fixture
def tiny_dataset(tmp_path):
    rng = np.random.default_rng(0)
    for cls in ("cat", "dog"):
        (tmp_path / cls).mkdir()
        for i in range(10):
            arr = rng.integers(0, 255, (20 + i, 30, 3), dtype=np.uint8)
            Image.fromarray(arr).save(tmp_path / cls / f"{cls}_{i}.png")
    # exact duplicate of a cat image, filed under dog
    (tmp_path / "dog" / "dup.png").write_bytes((tmp_path / "cat" / "cat_0.png").read_bytes())
    (tmp_path / "dog" / "notes.txt").write_text("not an image")
    return tmp_path


def test_index_dataset_classes_and_dedup(tiny_dataset):
    idx = index_dataset(tiny_dataset)
    assert idx.classes == ["cat", "dog"]
    assert len(idx.paths) == 20
    assert idx.duplicates_removed == 1
    assert "dog/notes.txt" not in idx.paths
    assert np.bincount(idx.labels).tolist() == [10, 10]


def test_index_dataset_does_not_modify_files(tiny_dataset):
    before = {p: p.read_bytes() for p in tiny_dataset.rglob("*") if p.is_file()}
    index_dataset(tiny_dataset)
    after = {p: p.read_bytes() for p in tiny_dataset.rglob("*") if p.is_file()}
    assert before == after


def test_stratified_split_is_disjoint_complete_and_balanced():
    labels = np.repeat(np.arange(4), 100)
    tr, va, te = stratified_split(labels, 0.15, 0.15, seed=1)
    assert len(set(tr) & set(va)) == len(set(tr) & set(te)) == len(set(va) & set(te)) == 0
    assert len(tr) + len(va) + len(te) == len(labels)
    for part in (tr, va, te):
        counts = np.bincount(labels[part], minlength=4)
        assert counts.max() - counts.min() <= 1


def test_stratified_split_is_deterministic():
    labels = np.repeat(np.arange(3), 50)
    a = stratified_split(labels, 0.2, 0.2, seed=7)
    b = stratified_split(labels, 0.2, 0.2, seed=7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_load_image_resizes_to_rgb_square(tiny_dataset):
    img = load_image(tiny_dataset / "cat" / "cat_3.png", 16)
    assert img.shape == (16, 16, 3) and img.dtype == np.uint8


def test_channel_stats_matches_numpy():
    x = np.random.default_rng(0).integers(0, 256, (300, 8, 8, 3), dtype=np.uint8)
    mean, std = channel_stats(x)
    ref = x.reshape(-1, 3) / 255.0
    np.testing.assert_allclose(mean, ref.mean(0), rtol=1e-5)
    np.testing.assert_allclose(std, ref.std(0), rtol=1e-4)


def test_normalizer_produces_nchw_standardised_tensor():
    x = np.full((2, 4, 4, 3), 255, dtype=np.uint8)
    norm = Normalizer(np.array([0.5, 0.5, 0.5]), np.array([0.25, 0.25, 0.25]))
    out = norm(norm.to_unit(x))
    assert out.shape == (2, 3, 4, 4)
    assert torch.allclose(out, torch.full_like(out, 2.0))


def test_augmenter_keeps_shape_and_range():
    x = torch.rand(8, 3, 32, 32)
    out = Augmenter(seed=0)(x)
    assert out.shape == x.shape
    assert out.min() >= 0 and out.max() <= 1
    assert not torch.equal(out, x)
