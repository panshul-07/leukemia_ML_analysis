"""Offline checks for the two CNN+GNN experiments; no epoch training."""

import ast
import importlib.util
import json
from pathlib import Path
import sys

import nbformat
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leukemia_osl.model import create_model

torch.set_num_threads(2)


def test_only_two_gnn_notebooks_remain():
    assert [path.name for path in sorted((ROOT / "notebooks").glob("*.ipynb"))] == [
        "03_hybrid_cnn_gnn.ipynb",
        "05_kaggle_cnn_gnn.ipynb",
    ]


def test_notebooks_are_clean_and_parse():
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                assert cell.execution_count is None
                assert cell.outputs == []
                ast.parse(cell.source)


@pytest.mark.parametrize(
    ("backbone", "image_size", "parameter_limit"),
    [
        ("efficientnet_b3", 300, 12_000_000),
        ("efficientnet_v2_s", 384, 23_000_000),
        ("inception_v3", 299, 26_000_000),
    ],
)
def test_cnn_gnn_forward_backward_and_checkpoint(
    tmp_path, backbone, image_size, parameter_limit
):
    model = create_model(
        2,
        name="hybrid_cnn_gnn",
        pretrained_backbone=False,
        cnn_backbone_name=backbone,
        embed_dim=128,
        gnn_layers=2,
        graph_neighbors=8,
    )
    inputs = torch.randn(2, 3, image_size, image_size)
    logits, features = model.forward_with_features(inputs)
    nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert logits.shape == (2, 2)
    assert features.shape == (2, 128)
    assert next(model.cnn_backbone.parameters()).grad is not None
    assert model.graph_layers[0].qkv.weight.grad is not None
    assert all(torch.isfinite(parameter.grad).all()
               for parameter in model.parameters() if parameter.grad is not None)
    assert sum(parameter.numel() for parameter in model.parameters()) < parameter_limit
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = create_model(2, cnn_backbone_name=backbone)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))


def test_factory_rejects_removed_models():
    with pytest.raises(ValueError, match="Only the hybrid_cnn_gnn"):
        create_model(2, name="hybrid_cnn_transformer")


def test_kaggle_split_is_patient_disjoint():
    script = ROOT / "scripts/run_kaggle_gnn_experiment.py"
    spec = importlib.util.spec_from_file_location("kaggle_runner", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    relative = Path(
        "data/raw/kaggle_leukemia_classification/kagglehub_cache/datasets/andrewmvd/"
        "leukemia-classification/versions/2/C-NMC_Leukemia"
    )
    root = ROOT / relative
    if not root.is_dir():
        root = ROOT.parent / relative
    if not root.is_dir():
        pytest.skip("C-NMC dataset is not available")
    development, test = module.load_dataset(root)
    train, val = module.fold_split(development, "fold_0")
    assert len(development) == 10_661
    assert len(test) == 1_867
    assert set(train.patient_id).isdisjoint(val.patient_id)
    assert set(train.patient_id).isdisjoint(test.patient_id)
    assert set(val.patient_id).isdisjoint(test.patient_id)


def test_patient_smoothing_and_threshold_use_group_context():
    script = ROOT / "scripts/run_kaggle_gnn_experiment.py"
    spec = importlib.util.spec_from_file_location("kaggle_calibration", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    probabilities = torch.tensor(
        [[0.8, 0.2], [0.4, 0.6], [0.3, 0.7], [0.1, 0.9]], dtype=torch.float32
    ).numpy()
    adjusted = module.patient_smooth(
        probabilities, ["healthy", "healthy", "leukemia", "leukemia"], 1.0
    )
    assert adjusted[0, 1] == pytest.approx(0.4)
    assert adjusted[1, 1] == pytest.approx(0.4)
    assert adjusted[2, 1] == pytest.approx(0.8)
    assert adjusted[3, 1] == pytest.approx(0.8)
    threshold = module.validation_threshold(adjusted, [0, 0, 1, 1])
    assert 0.4 < threshold <= 0.8


def test_se_inception_gnn_forward_path():
    model = create_model(
        2,
        pretrained_backbone=False,
        cnn_backbone_name="inception_v3",
        channel_attention=True,
    )
    logits = model(torch.randn(1, 3, 299, 299))
    assert logits.shape == (1, 2)
    assert model.channel_attention is True


def test_python_sources_parse():
    for path in [
        ROOT / "src/leukemia_osl/model.py",
        ROOT / "scripts/run_grouped_experiment.py",
        ROOT / "scripts/run_kaggle_gnn_experiment.py",
        ROOT / "src/leukemia_osl/training.py",
    ]:
        ast.parse(path.read_text())
