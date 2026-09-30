import json

from chain_checker.modifier.model.persistence import save_json, save_text


def test_save_text_writes_the_content_verbatim(tmp_path):
    save_text("out.txt", "hello\nworld", str(tmp_path))

    assert (tmp_path / "out.txt").read_text() == "hello\nworld"


def test_save_text_writes_non_ascii_content_as_utf8_regardless_of_host_locale(tmp_path):
    save_text("out.txt", "café ☕ prompt", str(tmp_path))

    assert (tmp_path / "out.txt").read_bytes() == "café ☕ prompt".encode()


def test_save_json_writes_indented_sorted_json(tmp_path):
    save_json("out.json", {"b": 1, "a": 2}, str(tmp_path))

    written = (tmp_path / "out.json").read_text()
    assert written == json.dumps({"a": 2, "b": 1}, indent=2, sort_keys=True)
    assert json.loads(written) == {"a": 2, "b": 1}


def test_save_json_overwrites_an_existing_file(tmp_path):
    save_json("out.json", {"first": True}, str(tmp_path))
    save_json("out.json", {"second": True}, str(tmp_path))

    assert json.loads((tmp_path / "out.json").read_text()) == {"second": True}
