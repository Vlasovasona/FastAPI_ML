import pytest

from ml.model import SentimentPrediction, load_model


@pytest.fixture(scope="function") # scope: вызов модели будет происходить отдельно для каждого теста, а не один раз для всего модуля/сессии
def model():
    # Load the model once for each test function
    return load_model()


@pytest.mark.parametrize( # передача входных данных для многократного выполнения теста с различными аргументами
    "text, expected_label",
    [
        ("очень плохо", "negative"), # каждый элемент кортежа - отдельный тест
        ("очень хорошо", "positive"),
        ("по-разному", "neutral"),
    ],
)
def test_sentiment(model, text: str, expected_label: str):
    """Test function"""
    model_pred = model(text)
    assert isinstance(model_pred, SentimentPrediction)
    assert model_pred.label == expected_label
