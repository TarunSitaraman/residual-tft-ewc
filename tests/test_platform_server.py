import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import platform_server as S  # noqa: E402


def test_figure_path_serves_only_pngs_inside_the_figures_folder(tmp_path):
    figs = tmp_path / "docs" / "figures" / "spark"
    figs.mkdir(parents=True)
    (figs / "ok.png").write_bytes(b"png")
    (figs / "notes.txt").write_text("x")
    (tmp_path / "secret.png").write_bytes(b"x")

    assert S.figure_path(tmp_path, "/figures/spark/ok.png") == (figs / "ok.png").resolve()
    assert S.figure_path(tmp_path, "/figures/spark/notes.txt") is None
    assert S.figure_path(tmp_path, "/figures/spark/missing.png") is None
    assert S.figure_path(tmp_path, "/figures/spark/../../../secret.png") is None
    assert S.figure_path(tmp_path, "/figures/spark/..%2F..%2F..%2Fsecret.png") is None


def test_fingerprint_changes_when_a_watched_file_changes(tmp_path):
    (tmp_path / "results").mkdir()
    f = tmp_path / "results" / "run.json"
    f.write_text("{}")
    before = S.fingerprint(tmp_path)
    assert S.fingerprint(tmp_path) == before

    f.write_text('{"a": 1}')
    assert S.fingerprint(tmp_path) != before
