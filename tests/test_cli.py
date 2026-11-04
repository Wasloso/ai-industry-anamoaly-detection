import json
import subprocess
import sys


def test_data_commands_run_from_config(data_config, tmp_path):
    config = tmp_path / "data.yaml"
    config.write_text(
        f"root: {data_config.root}\ncategory: bottle\n"
        f"manifest: {data_config.manifest}\nimage_size: 20\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "defect_detection", "data", "validate", "--config", str(config)],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["splits"]["test"] == {"defect": 1, "normal": 1}
    assert report["batch_shape"] == [8, 3, 20, 20]
    output = tmp_path / "preview.png"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "defect_detection",
            "data",
            "preview",
            "--config",
            str(config),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    assert output.is_file()


def test_missing_data_has_a_readable_error(tmp_path):
    config = tmp_path / "missing.yaml"
    config.write_text("root: missing\ncategory: bottle\nmanifest: manifest.json\n")
    result = subprocess.run(
        [sys.executable, "-m", "defect_detection", "data", "validate", "--config", str(config)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Expected train/good and test" in result.stderr
    assert "Traceback" not in result.stderr
