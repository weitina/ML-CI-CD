import numpy as np
from model import train_and_predict, get_accuracy

def test_predictions_not_none():
    """Test 1: Sprawdza, czy otrzymujemy jakąkolwiek predykcję"""
    preds, _ = train_and_predict()
    assert preds is not None, "Predictions should not be None."

def test_predictions_length():
    """Test 2: Sprawdza długość listy predykcji"""
    #czy długość predykcji jest większa od 0 i odpowiada próbkom testowym
    preds, y_test = train_and_predict()
    assert len(preds) > 0, "Długość predykcji musi być większa od 0."
    assert len(preds) == len(y_test), "Długość predykcji musi odpowiadać liczbie próbek testowych."

def test_predictions_value_range():
    """Test 3: Sprawdza zakres wartości (Iris: 0, 1, 2)."""
    #czy predykcje mieszczą się w spodziewanym zakresie klas
    preds, _ = train_and_predict()
    unique_preds = set(preds)
    expected_classes = {0, 1, 2}
    assert unique_preds.issubset(expected_classes), f"Nieoczekiwane klasy w predykcjach: {unique_preds}"

def test_model_accuracy():
    """Test 4: Sprawdza, czy model osiąga co najmniej 70% dokładności"""
    #czy dokładność >= 70%
    accuracy = get_accuracy()
    assert accuracy >= 0.7, f"Dokładność modelu ({accuracy}) jest niższa niż wymagane 70%."