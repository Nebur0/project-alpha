"""Il container deve conservare lo stato di ogni portafoglio paper."""

from pathlib import Path

from project_alpha.jobs import paper_configs

REPO = Path(__file__).resolve().parents[1]


def test_every_paper_log_directory_is_a_volume() -> None:
    compose = (REPO / "deploy" / "compose.yaml").read_text(encoding="utf-8")
    dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")

    for config in paper_configs(REPO):
        for path in (config.log.parent, config.state.parent):
            relative = path.relative_to(REPO).as_posix()
            assert f":/app/{relative}\n" in compose, f"{relative} non è un volume"
            assert relative in dockerfile, f"{relative} non creato nel Dockerfile"
        if config.ledger is not None:
            assert config.ledger.relative_to(REPO).as_posix().startswith("data/")
