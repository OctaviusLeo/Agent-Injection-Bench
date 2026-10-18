"""Unit tests for aib.multi_tool_probe with a fake sender and a fake server."""

from __future__ import annotations

from aib import multi_tool_probe as MP
from tests.fake_openai import FakeOpenAI, completion


def test_probe_counts_parsed_turns_and_failures_per_setting():
    def send(url, body):
        if body.get("parallel_tool_calls") is True:
            return 200, {"choices": [{"message": {"tool_calls": [{}, {}]}}]}
        return 500, {"error": {"message": "does not match the expected peg-native format"}}

    res = MP.probe("http://x", [], {"temperature": 0.7}, seeds=2, send=send)
    assert res["requests"] == 18 and res["parsed"] == 6
    assert res["per_parallel_tool_calls_setting"] == {"unset": {"http_500": 6}, "false": {"http_500": 6},
                                                     "true": {"parsed_2_calls": 6}}


def test_post_handles_success_and_http_errors():
    with FakeOpenAI([(200, completion(content="x"), 0), (500, {"error": {"message": "bad"}}, 0)]) as srv:
        assert MP.post(srv.base_url + "/chat/completions", {"a": 1})[0] == 200
        status, data = MP.post(srv.base_url + "/chat/completions", {"a": 1})
    assert status == 500 and data["error"]["message"] == "bad"


def test_main_writes_stamped_result(tmp_path, monkeypatch, capsys):
    import json
    import shutil

    import aib.repro
    import aib.server

    root = tmp_path / "repo"
    shutil.copytree(aib.repro.REPO_ROOT / "config", root / "config")
    (root / "results").mkdir()
    monkeypatch.setattr(aib.repro, "REPO_ROOT", root)
    monkeypatch.setattr(aib.repro, "require_clean", lambda: "7" * 40)
    monkeypatch.setattr(aib.repro, "make_stamp", lambda c, s: aib.repro.Stamp(c, s, "7" * 40))
    monkeypatch.setattr(aib.server, "http_get_json", lambda url: (200, {"build_info": "b1", "total_slots": 1}))
    seen = {}

    def fake_probe(url, tools, sampling, seeds):
        seen.update(url=url, n_tools=len(tools), seeds=seeds, sampling=sampling)
        return {"per_parallel_tool_calls_setting": {"unset": {"parsed_2_calls": 3}}, "requests": 3, "parsed": 3,
                "parse_rate": 1.0}

    monkeypatch.setattr(MP, "probe", fake_probe)
    assert MP.main(["--model", "qwen2.5-7b", "--seeds", "1"]) == 0
    saved = json.loads((root / "results" / "m0_multi_tool_probe_qwen2.5-7b_gguf-template.json").read_text())
    assert saved["parse_rate"] == 1.0 and saved["stamp"]["commit"] == "7" * 40 and saved["server"]["build_info"] == "b1"
    assert seen["n_tools"] == 24 and seen["seeds"] == 1 and seen["url"].endswith("/v1/chat/completions")
    assert '"parse_rate": 1.0' in capsys.readouterr().out
