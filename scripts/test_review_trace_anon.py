import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_trace_anon as review  # noqa: E402


class ReviewTraceAnonTests(unittest.TestCase):
    def test_only_changed_lines_with_context_and_keys_are_included(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for name in ['in', 'out', 'keys']:
                (root / name).mkdir()
            original = {'text': 'a\nb\nmail ada@example.org\nc\nd\ne\nf'}
            anonymized = {'text': 'a\nb\nmail x@example.com\nc\nd\ne\nf'}
            unchanged = '{"z":1}'
            (root / 'in/s.jsonl').write_text(json.dumps(original) + '\n' + unchanged + '\n')
            (root / 'out/s.jsonl').write_text(json.dumps(anonymized) + '\n' + unchanged + '\n')
            (root / 'keys/s.keys.jsonl').write_text(
                '{"version":"1"}\n{"original":"ada@example.org","replacement":"x@example.com","pattern_name":"email"}\n')
            sessions = review.build(root / 'in', root / 'out', root / 'keys')
            self.assertEqual(sessions, [{
                'name': 's.jsonl',
                'changes': [{'line': 1, 'path': 'text', 'rows': [
                    [[['b', False]], [['b', False]]],
                    [[['mail ', False], ['ada@example.org', True]],
                     [['mail ', False], ['x@example.com', True]]],
                    [[['c', False]], [['c', False]]]]}],
                'keys': [['ada@example.org', 'x@example.com', 'email']],
                'images_removed': 0,
                'classes': {'email': 1},
            }])

    def test_embedded_data_cannot_close_the_script_element(self):
        body = review.page([{'name': '</script><b>x', 'changes': [], 'keys': [], 'images_removed': 0, 'classes': {}}])
        self.assertEqual(body.count(b'</script>'), 2)


if __name__ == '__main__':
    unittest.main()
