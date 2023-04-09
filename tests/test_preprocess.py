import pytest
from typing import Tuple, Dict
import pandas as pd
import numpy as np

from preprocess import (
    load_raw_data,
    generate_negative_samples,
    convert2uij,
    load_train_data,
)


@pytest.fixture
def data() -> Tuple[pd.DataFrame, Dict[str, int]]:
    inter_data, data_info = load_raw_data()
    return inter_data, data_info


def test_load_raw_data(data: Tuple[pd.DataFrame, Dict[str, int]]):
    inter_data, data_info = data

    assert "users" in data_info
    assert "items" in data_info
    assert "ratings" in data_info
    assert inter_data.shape[1] == 2

    assert inter_data["user_id"].max() == data_info["users"]
    assert inter_data["item_id"].max() == data_info["items"]


def test_generate_negative_samples(data: Tuple[pd.DataFrame, Dict[str, int]]):
    inter_data, data_info = data
    inter_data, non_inter_data = generate_negative_samples(inter_data, data_info)
    assert (
        inter_data.apply(len) + non_inter_data.apply(len) == data_info["items"]
    ).all()


def test_convert2uij(data: Tuple[pd.DataFrame, Dict[str, int]]):
    inter_data, data_info = data
    inter_data, non_inter_data = generate_negative_samples(inter_data, data_info)

    combined_df = convert2uij(inter_data, non_inter_data, data_info)

    # 길이 확인
    assert combined_df.shape[0] == data_info["ratings"]
    assert combined_df.shape[1] == 3

    # 컬럼 이름 확인
    assert combined_df.columns[0] == "user_id"
    assert combined_df.columns[1] == "positive"
    assert combined_df.columns[2] == "negative"

    # 타입 확인
    assert isinstance(combined_df.iloc[5]["user_id"], np.integer)
    assert isinstance(combined_df.iloc[50]["positive"], np.integer)
    assert isinstance(combined_df.iloc[350]["negative"], np.ndarray)


def test_load_train_data():
    train_data, data_info = load_train_data()

    assert train_data["user_id"].max() == data_info["users"]
    assert train_data["positive"].max() == data_info["items"]
    assert len(train_data) == data_info["ratings"]
