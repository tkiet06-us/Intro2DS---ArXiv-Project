from src.schemas import Paper, Prediction


def test_paper_schema():
    assert Paper(arxiv_id="2506.00001").arxiv_id == "2506.00001"


def test_prediction_nullable():
    assert Prediction(arxiv_id="x", partition="test", citation_key="k").cand_1 is None
