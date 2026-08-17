from freqtrade.hedge.production.model_artifact_v2 import ModelArtifactRegistryV2, ModelArtifactV2


def test_model_artifact_v2_binds_full_promotion_lineage() -> None:
    artifact = ModelArtifactV2("m", "HPRL", *(str(i)*64 for i in range(8)))
    registry = ModelArtifactRegistryV2()
    assert registry.register(artifact) == artifact.fingerprint
    assert registry.get("m") is artifact
