# Uruchomienie projektu od zera

Polecenia wykonuj w katalogu głównym repozytorium **ai-industry-anamoaly-detection**, a nie w repozytorium notatek `isa-s2-projects-docs`.

## 1. Środowisko i szybkie sprawdzenie bez danych

Potrzebne są Git i [uv](https://docs.astral.sh/uv/). Projekt używa Pythona 3.12, wskazanego w `.python-version`. Na Linux i Windows pierwsza instalacja pobiera kilka GB zależności PyTorch/CUDA. Ta sama instalacja obsługuje pracę na CPU; zgodna karta NVIDIA jest opcjonalna.

```bash
git clone https://github.com/Wasloso/ai-industry-anamoaly-detection.git
cd ai-industry-anamoaly-detection
uv sync --locked
uv run --locked defect check-env
uv run --locked python scripts/check_patchcore.py
uv run --locked pytest -q
uv run --locked ruff check .
```

Test syntetyczny powinien wypisać informację o urządzeniu, liczbie obrazów i rozmiarze banku. Używa losowych wag i sprawdza wykonanie kodu, a nie jakość detekcji. Ten etap nie wymaga zbioru danych ani pobierania wytrenowanych wag ResNet.

Główna aplikacja korzysta z naszego PatchCore w PyTorch. Standardowa instalacja nie zawiera Anomalib.

## 2. Przygotowanie danych

Pobierz archiwum `Bottle` z [oficjalnej strony MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads) i rozpakuj do następującej struktury:

```text
data/mvtec_ad/bottle/
  train/good/*.png
  test/good/*.png
  test/broken_large/*.png
  test/<pozostale_typy_wad>/*.png
  ground_truth/broken_large/*_mask.png
  ground_truth/<pozostale_typy_wad>/*_mask.png
```

Dane, wagi, modele i wyniki nie są przesyłane przez Git. Każda osoba pobiera dane osobno. Ustawienia znajdują się w `configs/mvtec-bottle.yaml`; ścieżki `root` i `manifest` są liczone względem pliku YAML.

```bash
uv run --locked defect data validate --config configs/mvtec-bottle.yaml
uv run --locked defect data preview --config configs/mvtec-bottle.yaml
```

Pierwsze polecenie zapisuje manifest podziału i wypisuje jego podsumowanie. Drugie zapisuje `artifacts/preview.png` z obrazami i maskami referencyjnymi. To podgląd danych, nie predykcja modelu. Oficjalny test pozostaje oddzielony; poprawne obrazy treningowe są dzielone na dopasowanie, walidację i kalibrację.

## 3. Dopasowanie i kalibracja własnego PatchCore

```bash
uv run --locked defect train --config configs/mvtec-bottle-patchcore.yaml --run-id bottle-native-001
uv run --locked defect calibrate --artifact artifacts/bottle-native-001 --config configs/mvtec-bottle.yaml
```

Przy pierwszym dopasowaniu torchvision pobierze wagi ImageNet dla ResNet-18. Wymagany jest wtedy dostęp do internetu. Kolejne dopasowania mogą korzystać z cache. Lokalny cache można opcjonalnie ustawić przez zmienną `TORCH_HOME` wskazującą `.cache/torch`.

`train` buduje bank cech poprawnych produktów; nie optymalizuje wag ResNetu. Po nim katalog modelu zawiera `model.pt`, `metadata.json` i `manifest.json`. `calibrate` wyznacza progi na wydzielonych poprawnych obrazach i dopisuje `calibration.json`. Próg obrazu i próg mapy są osobne.

Domyślnie urządzenie wybierane jest automatycznie. Aby wymusić CPU, dopisz `--device cpu` do poleceń `train`, `calibrate` i `predict`. Do odczytu zapisanej wersji modelu nie potrzeba ponownego pobierania wag ani obrazów treningowych.

## 4. Predykcja i sprawdzenie wyników

```bash
uv run --locked defect predict --artifact artifacts/bottle-native-001 --input data/mvtec_ad/bottle/test/broken_large/000.png --output artifacts/demo-image-001
uv run --locked defect predict --artifact artifacts/bottle-native-001 --input data/mvtec_ad/bottle/test --output artifacts/demo-folder-001
```

Wynik podsumowujący pojawi się w terminalu. Katalog wyjściowy zawiera `results.json` oraz pliki dla każdego zdjęcia:

| Plik dla wejścia `000.png` | Zawartość |
| --- | --- |
| `000.png.map.npy` | Surowa mapa wyników anomalii |
| `000.png.heatmap.png` | Wizualizacja mapy |
| `000.png.mask.png` | Obszary przekraczające próg pikselowy |
| `000.png.overlay.png` | Podejrzane obszary zaznaczone na zdjęciu |

Przy analizie folderu zachowywane są jego podkatalogi. Wartość wyniku nie jest prawdopodobieństwem wady, a maska nie gwarantuje dokładnego obrysu uszkodzenia. Pełna ocena jakości jest kolejnym etapem prac.

## 5. Najczęstsze problemy

- **Brak plików danych:** sprawdź układ katalogów i `root` w YAML. Samo sklonowanie repozytorium nie pobiera MVTec AD.
- **Istniejący model lub wynik:** wybierz nowy `run-id` i nowe katalogi wyjściowe, np. z końcówką `002`. Polecenia nie nadpisują poprzednich uruchomień.
- **Zmienione dane lub podział:** wskaż nowy plik `manifest` w konfiguracji i dopasuj nowy model. Nie mieszaj kalibracji i modelu z różnych podziałów.
- **Artefakt Anomalib:** stary format 1 jest eksperymentalny. Główna ścieżka obsługuje własny model w formacie 2; wykonaj nowe `train` i `calibrate`.

## 6. Eksperymenty i dalsze prace

Dawne próby Anomalib są dostępne osobno:

```bash
uv run --locked --extra experiments python experiments/check_anomalib_models.py
```

To sprawdzenie architektur na losowych obrazach; nie jest potrzebne do uruchomienia głównej aplikacji. Może pobierać dodatkowe wagi. `uv sync --locked` przywraca standardowy zestaw zależności.

Obecnie dostępne są `check-env`, `data validate`, `data preview`, `train`, `calibrate` i `predict`. Ocena zbiorcza, API, dashboard i symulator pozostają w [roadmapie](implementation-roadmap.md); ich polecenia nie są jeszcze zaimplementowane. Następne zadanie to ocena konfiguracji jednego detektora, z ustalonym podziałem danych i pomiarami jakości, pamięci oraz czasu.
