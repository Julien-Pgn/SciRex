import tarfile

from scirex.ingestion.extract_latex import extract_latex


def _make_tarball(tmp_path, name, files):
    archive_path = tmp_path / f"{name}.tar.gz"
    src_dir = tmp_path / f"{name}_src"
    src_dir.mkdir()
    for filename, content in files.items():
        (src_dir / filename).write_text(content)
    with tarfile.open(archive_path, "w:gz") as tar:
        for filename in files:
            tar.add(src_dir / filename, arcname=filename)
    return archive_path


def test_extract_latex_returns_true_when_tex_file_present(tmp_path):
    archive = _make_tarball(tmp_path, "with_tex", {"paper.tex": "\\documentclass{article}"})
    output_dir = tmp_path / "out"
    assert extract_latex("1111.1111", archive, output_dir) is True
    assert (output_dir / "1111.1111" / "paper.tex").exists()


def test_extract_latex_returns_false_when_no_tex_file(tmp_path):
    archive = _make_tarball(tmp_path, "without_tex", {"readme.txt": "no latex here"})
    output_dir = tmp_path / "out"
    assert extract_latex("2222.2222", archive, output_dir) is False
    assert (output_dir / "2222.2222" / "readme.txt").exists()
