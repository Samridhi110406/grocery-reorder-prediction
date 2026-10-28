import pandas as pd
import pytest

from src.evaluate import evaluate, per_user_recall

@pytest.fixture
def toy_data():
    rows = []
    for rank in range(1,13):
        rows.append({"user_id":1, "rank":rank, "label": int(rank in (1,3,12))})
    for rank in range(1,4):
        rows.append({"user_id":2, "rank": rank, "label": int(rank==1)})
    for rank in range(1,3):
        rows.append({"user_id":3, "rank":rank, "label":0})
    return pd.DataFrame(rows)

def test_users_with_no_positives_are_excluded(toy_data):
    result = evaluate(toy_data, "rank")
    assert result.loc[0, "Users"] == 2

def test_recall(toy_data):
    result = evaluate(toy_data, "rank")
    assert result.loc[0, "recall_at_5"] == pytest.approx((2/3+1)/2)
    assert result.loc[0, "recall_at_10"] == pytest.approx((2/3+1)/2)

def test_precision_uses_min_k_candidates(toy_data):
    result = evaluate(toy_data,"rank")
    assert result.loc[0, "precision_at_5"] == pytest.approx((2/5+1/3)/2)
    assert result.loc[0, "precision_at_10"] == pytest.approx((2/10+1/3)/2)

def test_per_user_recall_matches_evaluate(toy_data):
    per_user = per_user_recall(toy_data, "rank", k=10)
    assert per_user.mean() == pytest.approx(evaluate(toy_data, "rank").loc[0, "recall_at_10"])