from pathlib import Path

from atropos.quality.calibrate import fit_affine_calibration, run_calibration
from atropos.quality.predictor import QualityPredictorConfig, predict_quality_degradation
from atropos.quality.sensitivity import LayerSensitivity, SensitivityProfile


def test_quality_predictor_methods() -> None:
    profile = SensitivityProfile(
        layers=(
            LayerSensitivity("l0", gradient_magnitude=0.4, hessian_trace=0.5),
            LayerSensitivity("l1", gradient_magnitude=0.3, hessian_trace=0.2),
        )
    )

    linear = predict_quality_degradation(
        metric="perplexity",
        sparsity=0.3,
        sensitivity_profile=profile,
        baseline_quality=1.0,
        predictor_config=QualityPredictorConfig(method="linear"),
    )
    obd = predict_quality_degradation(
        metric="perplexity",
        sparsity=0.3,
        sensitivity_profile=profile,
        baseline_quality=1.0,
        predictor_config=QualityPredictorConfig(method="obd_obs"),
    )
    lookup = predict_quality_degradation(
        metric="perplexity",
        sparsity=0.3,
        sensitivity_profile=profile,
        baseline_quality=1.0,
        predictor_config=QualityPredictorConfig(method="lookup", lookup_table={0.3: 12.0}),
    )

    assert linear.degradation_percent > obd.degradation_percent
    assert lookup.degradation_percent == 12.0
    assert 0 <= linear.expected_quality <= 1.0


def test_fit_affine_calibration() -> None:
    fit = fit_affine_calibration([(10.0, 9.0), (20.0, 17.0), (30.0, 25.0)])
    assert fit.num_samples == 3
    assert fit.scale > 0


def test_run_calibration_no_data(tmp_path: Path) -> None:
    validation_dir = tmp_path / "validation_results"
    validation_dir.mkdir()
    out_path = tmp_path / "quality_calibration.yaml"
    fit = run_calibration(validation_dir=validation_dir, output_path=out_path)
    assert fit.num_samples == 0
    assert out_path.exists()
