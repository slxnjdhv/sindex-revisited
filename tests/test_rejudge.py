import json

from hallu.labels import parse_yes_no
from hallu.prompts import judge_prompt
from hallu.rejudge import find_runs, merge


def test_parse_yes_no():
    assert parse_yes_no("Yes") is True
    assert parse_yes_no("**Yes**") is True          # failed with startswith("yes")
    assert parse_yes_no("Response: yes.") is True   # failed with startswith("yes")
    assert parse_yes_no('"No"') is False
    assert parse_yes_no("no, it is Paris") is False
    assert parse_yes_no("Yesterday") is None        # not a standalone yes
    assert parse_yes_no("") is None


def test_judge_prompt_v2_mentions_extra_words():
    p = judge_prompt("Q?", ["Boots"], "Dora's monkey is named Boots.")
    assert "Extra words" in p and "Boots" in p


def _w(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_merge_majority_and_consensus(tmp_path):
    run = tmp_path / "m_triviaqa_n400_P10"
    run.mkdir()
    _w(run / "generations.jsonl", [{"id": i} for i in "abc"])
    _w(run / "labels.jsonl", [
        {"id": "a", "pred": "x", "gold": ["x"], "ensemble": 1, "embedding": 1},
        {"id": "b", "pred": "y", "gold": ["y"], "ensemble": 0, "embedding": 1},
        {"id": "c", "pred": "z", "gold": ["z"], "ensemble": 1, "embedding": 1},
        {"id": "c", "pred": "z", "gold": ["z"], "ensemble": 1, "embedding": 1},  # duplicate id
    ])
    _w(run / "judge_j1.jsonl", [{"id": "a", "label": 1}, {"id": "b", "label": 1}, {"id": "c", "label": 0}])
    _w(run / "judge_j2.jsonl", [{"id": "a", "label": 1}, {"id": "b", "label": 0}, {"id": "c", "label": 0}])
    _w(run / "judge_j3.jsonl", [{"id": "a", "label": 0}, {"id": "b", "label": 1}, {"id": "c", "label": None}])
    assert find_runs(tmp_path, "*_n400_P10") == [run]
    merge([run], ["j1", "j2", "j3"])
    rows = {r["id"]: r for r in map(json.loads, open(run / "labels.jsonl"))}
    assert len(rows) == 3
    assert rows["a"]["judges_majority"] == 1 and rows["a"]["consensus"] == 1
    assert rows["b"]["judges_majority"] == 1 and rows["b"]["consensus"] == 0   # ensemble says correct
    assert rows["c"]["judges_majority"] == 0 and rows["c"]["judge_j3"] is None  # 2 votes, both correct
