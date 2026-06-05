import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _copy_git_tracked_files(destination: Path) -> None:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    )
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative_path = Path(raw_path.decode())
        source = ROOT / relative_path
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _assert_transfer_bundle_exists(root: Path) -> None:
    transfer = root / "transfer"
    required_transfer_paths = [
        transfer / "wheels",
        transfer / "models",
        transfer / "src" / "requirements.frozen.txt",
    ]
    missing = [str(path.relative_to(root)) for path in required_transfer_paths if not path.exists()]
    assert missing == []
    assert any((transfer / "wheels").iterdir())
    assert any((transfer / "models").iterdir())


def test_offline_docker_compose_deployment_workflow(tmp_path):
    if shutil.which("docker") is None:
        pytest.skip("Docker is not installed")

    transfer = ROOT / "transfer"
    required_transfer_paths = [
        transfer / "wheels",
        transfer / "models",
        transfer / "src" / "requirements.frozen.txt",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required_transfer_paths if not path.exists()]
    if missing:
        pytest.skip("offline transfer bundle missing: " + ", ".join(missing))

    _assert_transfer_bundle_exists(ROOT)

    context = tmp_path / "context"
    context.mkdir()
    _copy_git_tracked_files(context)
    shutil.copytree(transfer, context / "transfer")

    project = f"rag-mcp-offline-test-{uuid.uuid4().hex[:12]}"
    env = os.environ | {"DATA_DIR": str(tmp_path / "data")}
    (tmp_path / "data").mkdir()
    compose = [
        "docker",
        "compose",
        "-p",
        project,
        "-f",
        "docker-compose.yaml",
        "-f",
        "docker-compose.offline.yaml",
    ]

    try:
        subprocess.run(compose + ["build"], cwd=context, env=env, check=True)
        subprocess.run(
            compose
            + [
                "run",
                "--rm",
                "--no-deps",
                "rag-mcp",
                "python",
                "-c",
                "import chunkers, dashboard, server, store; assert store.MODEL_NAME",
            ],
            cwd=context,
            env=env,
            check=True,
        )
    finally:
        subprocess.run(
            compose + ["down", "--volumes", "--remove-orphans"],
            cwd=context,
            env=env,
            check=False,
        )


def test_prepare_transfer_docker_creates_offline_deployable_bundle(tmp_path):
    if os.environ.get("RUN_OFFLINE_TRANSFER_TEST") != "1":
        pytest.skip("set RUN_OFFLINE_TRANSFER_TEST=1 to run slow networked Docker test")
    if shutil.which("docker") is None:
        pytest.skip("Docker is not installed")

    context = tmp_path / "context"
    context.mkdir()
    _copy_git_tracked_files(context)

    subprocess.run(["bash", "prepare-transfer-docker.sh"], cwd=context, check=True)
    _assert_transfer_bundle_exists(context)

    project = f"rag-mcp-generated-offline-test-{uuid.uuid4().hex[:12]}"
    env = os.environ | {"DATA_DIR": str(tmp_path / "data")}
    (tmp_path / "data").mkdir()
    compose = [
        "docker",
        "compose",
        "-p",
        project,
        "-f",
        "docker-compose.yaml",
        "-f",
        "docker-compose.offline.yaml",
    ]

    try:
        subprocess.run(compose + ["build"], cwd=context, env=env, check=True)
        subprocess.run(
            compose
            + [
                "run",
                "--rm",
                "--no-deps",
                "rag-mcp",
                "python",
                "-c",
                "import chunkers, dashboard, server, store; assert store.MODEL_NAME",
            ],
            cwd=context,
            env=env,
            check=True,
        )
    finally:
        subprocess.run(
            compose + ["down", "--volumes", "--remove-orphans"],
            cwd=context,
            env=env,
            check=False,
        )
