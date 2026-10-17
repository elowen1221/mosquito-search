"""评测数据集加载单元测试。"""

import json

import pytest

from mosquito.evaluation.dataset import DatasetError, load_judgments


def _write(tmp_path, lines):
    path = tmp_path / "judgments.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_valid(tmp_path):
    path = _write(
        tmp_path,
        [
            json.dumps({"query": "python", "judgments": {"a/b": 2}}),
            json.dumps({"query": "rust", "judgments": {"c/d": 1}}),
        ],
    )
    judgments = load_judgments(path)
    assert len(judgments) == 2
    assert judgments[0].query == "python"
    assert judgments[1].judgments == {"c/d": 1}


def test_skips_blank_lines(tmp_path):
    path = _write(tmp_path, ["", json.dumps({"query": "python", "judgments": {"a/b": 2}}), ""])
    assert len(load_judgments(path)) == 1


def test_malformed_json(tmp_path):
    path = _write(tmp_path, ["{not valid json"])
    with pytest.raises(DatasetError, match="第 1 行"):
        load_judgments(path)


def test_invalid_judgment_value(tmp_path):
    path = _write(tmp_path, [json.dumps({"query": "python", "judgments": {"a/b": 5}})])
    with pytest.raises(DatasetError, match="第 1 行"):
        load_judgments(path)


def test_missing_file(tmp_path):
    with pytest.raises(DatasetError, match="不存在"):
        load_judgments(tmp_path / "nope.jsonl")