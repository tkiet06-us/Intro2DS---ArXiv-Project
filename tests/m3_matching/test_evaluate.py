from src.m3_matching.evaluate import mrr


def test_mrr():
    assert mrr([["a", "b"], ["x", "y"]], ["b", "z"]) == 0.25
