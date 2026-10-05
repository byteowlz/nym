#!/usr/bin/env python3
"""Deterministic, handwritten synthetic corpus. No detector/model/network labeling.

[[class:scope|value]] labels a sensitive occurrence; [[benign|value]] labels
an approved literal. Markers are removed before UTF-8 offsets are computed.
"""
import json
from pathlib import Path
import re

OUT = Path(__file__).with_name("fixtures") / "agent_traces.json"
MARK = re.compile(r"\[\[([^|]+)\|([^\]]+)\]\]")


def marked(text, path, gold, benign):
    output, cursor = "", 0
    for match in MARK.finditer(text):
        output += text[cursor:match.start()]
        value = match[2]
        row = {"path": path, "start": len(output.encode()),
               "end": len((output + value).encode()), "value": value}
        if match[1] == "benign":
            benign.append(row)
        else:
            cls, scope = match[1].split(":")
            gold.append({**row, "class": cls, "scope": scope})
        output += value
        cursor = match.end()
    return output + text[cursor:]


def build_case(index, split, name, email, account, pin, dob, codename):
    gold, benign = [], []
    uid = f"00000000-0000-4000-8000-{index:012d}"
    records = []

    def record(message):
        number = len(records)
        records.append({"id": f"trace-{index}-{number}",
                        "parentId": f"trace-{index}-{number-1}" if number else uid,
                        "timestamp": f"2026-02-0{index}T10:20:30.000Z",
                        "type": "message", "message": message})

    record({"role": "user", "content": [{"type": "text", "text":
        f"This is a fictional test task for [[person:ner|{name}]]. Contact [[email:regex|{email}]]. "
        "Keep [[benign|--strict]], [[benign|--timeout 30]], and [[benign|--no-progress]] in the CLI. "
        "Public dependency [[benign|serde_json]] is at [[benign|https://docs.rs/serde_json/]]."}]})
    record({"role": "assistant", "content": [{"type": "thinking", "thinking":
        "Reasoning: use [[benign|bash]] and [[benign|cargo check]]. The buffer size [[benign|4096]] "
        "is not a PIN. The build timestamp [[benign|2026-02-01T10:20:30Z]] is not a birthday. "
        f"Fictional account username=[[username:regex|{account}]]; do not confuse [[benign|user_count]] "
        "with an account. Public [[benign|tokio]] and [[benign|clap]] are dependencies."},
        {"type": "toolCall", "id": f"call-{index}", "name": "[[benign|bash]]", "arguments": {
            "command": "[[benign|rg --files src]] && [[benign|cargo check --offline]]",
            "cwd": "[[benign|/tmp/synthetic-project]]", "timeout": 30, "parallel": False}}],
        "usage": {"input": 4096, "output": 128, "cacheRead": 0, "cost": 0.0125},
        "stopReason": "toolUse"})
    record({"role": "toolResult", "toolCallId": f"call-{index}", "toolName": "[[benign|bash]]",
            "content": [{"type": "text", "text":
                "[[benign|src/lib.rs]]\n[[benign|warning: unused import]]\n"
                "[[benign|0 errors]]; local listener [[benign|127.0.0.1:8080]] and [[benign|::1]]. "
                "Compiler cache [[benign|/usr/local/lib]] is public fixture layout."}],
            "isError": False, "details": {"exitCode": 0, "truncated": False}})
    record({"role": "assistant", "content": [{"type": "text", "text":
        "Proposed source:\n[[benign|let buffer_size: usize = 4096;]]\n"
        "[[benign|const LIMIT: u32 = 8080;]]\n[[benign|fn main() { println!(\"ready\"); }]]\n"
        f"Repeat contact [[email:regex|{email}]] and [[email:regex|{email}]] for occurrence accounting. "
        f"Patient [[person:ner|{name}]] has date of birth [[birth_date:ner|{dob}]]; "
        f"the authentication PIN is [[pin:ner|{pin}]]. "
        "Never use these fabricated credentials: [[aws_key:regex|AKIAIOSFODNN7EXAMPLE]] and "
        "[[api_key:regex|sk_test_syntheticOnlyNotARealKey123456789]]."}]})
    record({"role": "user", "content": [{"type": "text", "text":
        f"Known-sensitive fictional project [[codename:policy|{codename}]] uses "
        f"[[internal_hostname:policy|{codename.lower()}.svc.example.invalid]]. "
        "These require an explicit known-sensitive policy, not a universal NER promise."}]})
    trace = {"session": {"id": uid, "version": 3, "type": "session"}, "records": records,
             "metadata": {"identityId": "[[identity_uuid:regex|10000000-0000-4000-8000-000000000099]]",
                          "contact": f"[[email:regex|{email}]]", "reviewed": None}}

    def walk(value, path=""):
        if isinstance(value, dict):
            return {key: walk(item, f"{path}.{key}" if path else key) for key, item in value.items()}
        if isinstance(value, list):
            return [walk(item, f"{path}[{i}]") for i, item in enumerate(value)]
        return marked(value, path, gold, benign) if isinstance(value, str) else value

    return {"id": f"synthetic-{index}", "split": split, "trace": walk(trace),
            "gold": gold, "benign": benign}


def document():
    return {"version": "agent-traces-v1", "origin": "handwritten-synthetic-only",
            "cases": [
                build_case(1, "selection", "Mira Quenwick", "mira@example.invalid", "scribe_amber", "4096", "1992-07-16", "AmberKestrel"),
                build_case(2, "selection", "Jörg Vexlorn", "jorg@example.invalid", "scribe_violet", "5729", "1987-03-24", "VioletHeron"),
                build_case(3, "holdout", "Zoë Nimbrel", "zoe@example.invalid", "scribe_cobalt", "7318", "1995-11-08", "CobaltFinch"),
                build_case(4, "holdout", "Renée Talvex", "renee@example.invalid", "scribe_copper", "2647", "1989-06-19", "CopperWren"),
            ]}


if __name__ == "__main__":
    OUT.write_text(json.dumps(document(), ensure_ascii=False, indent=2) + "\n")
