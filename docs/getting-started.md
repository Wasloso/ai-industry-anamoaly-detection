# Uruchomienie wstępnej wersji projektu

Wykonuj polecenia w repozytorium `ai-industry-anamoaly-detection`, nie w repozytorium notatek.

## 1. Środowisko i konfiguracje

Potrzebne są Git i [uv](https://docs.astral.sh/uv/). Wersja Pythona 3.12 jest wskazana w `.python-version`. Na Linux i Windows pierwsza instalacja pobiera także duże pakiety CUDA; karta NVIDIA nie jest wymagana do sprawdzenia modeli na CPU.

```bash
git clone https://github.com/Wasloso/ai-industry-anamoaly-detection.git
cd ai-industry-anamoaly-detection
uv sync --locked
uv run --locked defect check-env
uv run --locked defect model check --config configs/mvtec-bottle-patchcore.yaml
uv run --locked defect model check --config configs/mvtec-bottle-padim.yaml
```

Oczekiwane klasy to `Patchcore` i `Padim`. Polecenia wczytują YAML i tworzą modele Anomalib z losowymi wagami, bez dopasowania i pobierania wag ImageNet. `fitted: false` jest prawidłowym wynikiem. Ten etap nie wymaga MVTec AD i nie generuje wyników detekcji.

W konfiguracjach ustawiono wspólny ResNet-18, warstwy `layer2`/`layer3` i rozmiar wejścia 256 × 256. PatchCore ma własne parametry coresetu i sąsiadów, a PaDiM liczbę modelowanych cech. `src/defect_detection/models/anomalib_models.py` ustawia wspólne skalowanie i normalizację; nie należy normalizować obrazów drugi raz w loaderze.

```bash
uv run --locked pytest -q
uv run --locked ruff check .
```

## 2. Opcjonalne sprawdzenie danych

Pobierz archiwum `Bottle` z [oficjalnej strony MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads). Oczekiwany układ:

```text
data/mvtec_ad/bottle/
  train/good/*.png
  test/good/*.png
  test/<typ_wady>/*.png
  ground_truth/<typ_wady>/*_mask.png
```

```bash
uv run --locked defect data validate --config configs/mvtec-bottle.yaml
uv run --locked defect data preview --config configs/mvtec-bottle.yaml
```

Pierwsze polecenie zapisuje manifest i wypisuje podziały danych, drugie tworzy `artifacts/preview.png`. To podgląd danych referencyjnych. Dane, cache, modele i wyniki nie są dostarczane przez Git. Ścieżki w YAML są względne wobec pliku konfiguracji.

Zmiana danych lub ustawień podziału wymaga nowego pliku `manifest`. Docelowo obie metody mają korzystać z tych samych podziałów do dopasowania, walidacji, kalibracji i końcowej oceny.

## 3. Wcześniejszy eksperyment

Starszy własny PatchCore pozostaje dostępny oddzielnie:

```bash
uv run --locked defect native train --config experiments/configs/mvtec-bottle-native.yaml --run-id native-example-001
uv run --locked defect native calibrate --artifact artifacts/native-example-001 --config configs/mvtec-bottle.yaml
uv run --locked defect native predict --artifact artifacts/native-example-001 --input data/mvtec_ad/bottle/test/broken_large/000.png --output artifacts/native-example-prediction-001
```

To nie jest uruchomienie modelu Anomalib. Ten eksperyment wymaga danych i przy pierwszym dopasowaniu może pobrać wagi ResNet. Wybieraj nowe nazwy modeli i katalogów wynikowych. Jego artefakty w formacie 2 oraz starsze pliki Anomalib w formacie 1 nie są formatem nowej ścieżki, która dopiero powstanie. Nie należy ich automatycznie konwertować ani mieszać.

## 4. Kolejne etapy

Następne zadania to dopasowanie i zapis obu modeli bibliotecznych, kalibracja, wspólna analiza oraz porównanie jakości i zasobów. Dopiero potem powstanie aplikacja Streamlit do wyboru obrazów, wywołania analizy i pobrania CSV/JSON oraz wizualizacji. Streamlit jest już zależnością, ale aplikacja nie jest jeszcze zaimplementowana.

Zakres i kolejność opisuje [roadmapa](implementation-roadmap.md). Projekt nie obejmuje strumieniowania, symulacji zdarzeń ani osobnego serwera API.
