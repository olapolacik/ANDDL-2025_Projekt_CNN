# ANDDL-2025 – Projekt CNN

## Temat
Analiza wpływu architektury i hiperparametrów sieci konwolucyjnej na klasyfikację obrazów Fashion-MNIST.

## Opis danych
- Dataset: Fashion-MNIST
- 70 000 obrazów 28x28 w skali szarości
- 10 klas (odzież, buty itp.)
- Podział: 60 000 treningowych, 10 000 testowych
- Normalizacja do zakresu [-1,1]

## Architektura
- CNN:
  - Conv2D → ReLU → MaxPool
  - Conv2D → ReLU → MaxPool
  - Flatten → FC → ReLU → FC → Softmax
- Funkcja aktywacji: ReLU lub LeakyReLU
- Optymalizator: Adam
- Loss: CrossEntropy

## Eksperymenty
1. Różne learning rate: 0.001 vs 0.0001
2. Różne głębokości modelu: 1 Conv vs 2 Conv
3. Różne funkcje aktywacji: ReLU vs LeakyReLU

## Wyniki
- Dokładność na zbiorze testowym: ~0.90–0.92 (w zależności od eksperymentu)
- Krzywe loss i accuracy zapisane w TensorBoard: `results/runs/ANDDL2025`

## Wizualizacje
- Błędne predykcje można wyświetlić za pomocą `train.py`
- Możliwe rozszerzenia: wizualizacja map cech (feature maps), augmentacja danych

## Uruchomienie
1. Zainstaluj pakiety:
```bash
pip install -r requirements.txt
