from app.tasks.common import write_text


def test_write_text_keeps_line_endings(tmp_path):
    path = write_text(tmp_path / "nested" / "table.csv", "a,b\r\n1,2\r\n")
    assert path.read_bytes() == b"a,b\r\n1,2\r\n"
