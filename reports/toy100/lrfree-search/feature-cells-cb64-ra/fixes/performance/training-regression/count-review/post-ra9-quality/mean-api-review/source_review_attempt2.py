"""Retain first AST-parser failure; resolve its literal starred scalar-key set."""
from pathlib import Path

original = Path(__file__).with_name('source_review.py')
text = original.read_text()
needle = 'actual_keys = ast.literal_eval(node.value)'
assert text.count(needle) == 1
replacement = '''scalar_assignment = next(item for item in mean_tree.body
                if isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SCALAR_FIELDS' for t in item.targets))
            scalar_fields = ast.literal_eval(scalar_assignment.value)
            assert isinstance(node.value, ast.Set)
            actual_keys = set()
            for field in node.value.elts:
                if isinstance(field, ast.Starred):
                    assert isinstance(field.value, ast.Name) and field.value.id == 'SCALAR_FIELDS'
                    actual_keys.update(scalar_fields)
                else:
                    actual_keys.add(ast.literal_eval(field))'''
text = text.replace(needle, replacement)
needle = "protected = dict(seal['source_and_input_sha256'])"
assert text.count(needle) == 1
text = text.replace(needle, needle + "\n    protected[str(HERE / 'source_review.py')] = sha(HERE / 'source_review.py')\n    protected[str(HERE / 'source-review.log')] = sha(HERE / 'source-review.log')")
exec(compile(text, str(original), 'exec'), {'__name__': '__main__', '__file__': __file__})
