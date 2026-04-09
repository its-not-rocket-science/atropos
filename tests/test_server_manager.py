import pytest

from atroposlib.envs.server_handling import ServerManager, ServerManagerError


def test_localhost_mode_defaults_are_used() -> None:
    manager = ServerManager()

    config = manager.from_environment({})

    assert config.mode == "localhost"
    assert config.host == "127.0.0.1"
    assert config.port == 8000
    assert config.world_size == 1
    assert config.rank == 0
    assert config.local_rank == 0
    assert config.node_count == 1


def test_localhost_mode_validates_server_port_integer() -> None:
    manager = ServerManager()

    with pytest.raises(ServerManagerError, match="SERVER_PORT must be an integer"):
        manager.from_environment({"SERVER_PORT": "not-a-number"})


def test_slurm_mode_single_node_is_parsed() -> None:
    manager = ServerManager()
    env = {
        "CLUSTER_LAUNCH_MODE": "slurm",
        "SLURM_NTASKS": "4",
        "SLURM_PROCID": "2",
        "SLURM_LOCALID": "0",
        "SLURM_JOB_NUM_NODES": "1",
        "MASTER_ADDR": "10.0.0.8",
        "MASTER_PORT": "29500",
    }

    config = manager.from_environment(env)

    assert config.mode == "slurm"
    assert config.host == "10.0.0.8"
    assert config.port == 29500
    assert config.world_size == 4
    assert config.rank == 2
    assert config.local_rank == 0
    assert config.node_count == 1


def test_slurm_mode_multi_node_rejects_localhost_master_addr() -> None:
    manager = ServerManager()
    env = {
        "CLUSTER_LAUNCH_MODE": "slurm",
        "SLURM_NTASKS": "8",
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
        "SLURM_JOB_NUM_NODES": "2",
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "29500",
    }

    with pytest.raises(ServerManagerError, match="routable"):
        manager.from_environment(env)


def test_slurm_mode_detected_from_environment_without_explicit_mode() -> None:
    manager = ServerManager()
    env = {
        "SLURM_NTASKS": "2",
        "SLURM_PROCID": "1",
        "MASTER_ADDR": "node0.cluster",
        "MASTER_PORT": "23456",
    }

    config = manager.from_environment(env)

    assert config.mode == "slurm"
    assert config.world_size == 2
    assert config.rank == 1


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("SLURM_NTASKS", "-1", "must be >= 1"),
        ("SLURM_PROCID", "NaN", "must be an integer"),
        ("MASTER_PORT", "99999", "must be <= 65535"),
        ("MASTER_ADDR", "", "Missing required"),
    ],
)
def test_slurm_mode_rejects_malformed_or_missing_values(
    key: str,
    value: str,
    message: str,
) -> None:
    manager = ServerManager()
    env = {
        "CLUSTER_LAUNCH_MODE": "slurm",
        "SLURM_NTASKS": "4",
        "SLURM_PROCID": "1",
        "MASTER_ADDR": "node0.cluster",
        "MASTER_PORT": "29500",
    }
    env[key] = value

    with pytest.raises(ServerManagerError, match=message):
        manager.from_environment(env)
