"""Test VCKOArtifact."""

import numpy as np

from vcko.artifact import VCKOArtifact, create_commitment_hash


def test_create_commitment_hash():
    coefs = np.array([0.1, 0.2, 0.3])
    intercept = 0.5
    means = np.array([1.0, 2.0, 3.0])
    stds = np.array([0.5, 0.5, 0.5])
    n = 100
    rate = 0.35

    hash1 = create_commitment_hash(coefs, intercept, means, stds, n, rate)
    hash2 = create_commitment_hash(coefs, intercept, means, stds, n, rate)

    assert hash1 == hash2
    assert len(hash1) == 64


def test_vcko_artifact_verify():
    artifact = VCKOArtifact(
        centre_id="test",
        feature_names=["f1", "f2"],
        coefficients=[0.1, 0.2],
        intercept=0.5,
        feature_means=[1.0, 2.0],
        feature_stds=[0.5, 0.5],
        n_samples=100,
        outcome_rate=0.35,
        commitment_hash=create_commitment_hash(
            np.array([0.1, 0.2]), 0.5, np.array([1.0, 2.0]), np.array([0.5, 0.5]), 100, 0.35
        ),
        metadata={},
    )

    assert artifact.verify()


def test_vcko_artifact_save_load(tmp_path):
    artifact = VCKOArtifact(
        centre_id="test",
        feature_names=["f1"],
        coefficients=[0.1],
        intercept=0.5,
        feature_means=[1.0],
        feature_stds=[0.5],
        n_samples=100,
        outcome_rate=0.35,
        commitment_hash="abc123",
        metadata={},
    )

    path = tmp_path / "test.json"
    artifact.save(path)

    loaded = VCKOArtifact.load(path)
    assert loaded.centre_id == artifact.centre_id
    assert loaded.coefficients == artifact.coefficients
