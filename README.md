# Wykrywanie defektów produktów na obrazach

Projekt na przedmiot **Algorytmy sztucznej inteligencji w Przemyśle 4.0**.

Porównujemy PatchCore i PaDiM z biblioteki Anomalib. Docelowo będzie to prosta
aplikacja Streamlit: użytkownik wskaże obrazy, uruchomi analizę, obejrzy wyniki
obu metod i pobierze zestawienie oraz wizualizacje. Analiza odbywa się na żądanie.

## Obecny etap

- Zależności Anomalib, PyTorch i Streamlit oraz lockfile.
- Wstępne konfiguracje obu modeli i wspólnego przygotowania obrazów.
- Sprawdzenie konstrukcji modeli bez uczenia i pobierania wag.
- Loader MVTec AD, trwały podział danych i podgląd obrazów/masek.

Dopasowanie i zapis modeli Anomalib, ich porównanie oraz aplikacja z eksportem
są kolejnymi krokami. Nie ma jeszcze polecenia analizy dla nowych modeli ani
gotowej aplikacji. Wcześniejszy własny PatchCore pozostał jako eksperyment.

## Uruchomienie

Potrzebne są Python 3.12 i [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
uv run --locked defect check-env
uv run --locked defect model check --config configs/mvtec-bottle-patchcore.yaml
uv run --locked defect model check --config configs/mvtec-bottle-padim.yaml
uv run --locked pytest -q
uv run --locked ruff check .
```

Sprawdzenie modeli nie wymaga danych i używa losowych wag. Wypisuje
`pretrained_weights: false` i `fitted: false`; nie jest oceną jakości.
PyTorch jest przypięty do wersji CUDA 12.6 na Linux i Windows i działa również
na CPU. Pierwsza instalacja może pobrać kilka GB zależności.

## Konfiguracje

| Plik | Zastosowanie |
| --- | --- |
| `configs/mvtec-bottle.yaml` | Ścieżki danych, rozmiar 256 × 256, podziały i seed |
| `configs/mvtec-bottle-patchcore.yaml` | ResNet-18, warstwy 2/3, coreset 1%, 9 sąsiadów |
| `configs/mvtec-bottle-padim.yaml` | ResNet-18, warstwy 2/3, 100 cech do modelowania rozkładu |

Oba modele mają wspólny preprocessing: skalowanie bez wycinania środka
i normalizację ImageNet. Domyślny postprocessing Anomalib jest wyłączony;
kalibracja progów na wydzielonych poprawnych obrazach wymaga jeszcze integracji.
`build_model` wymaga jawnego wyboru `pretrained`; przyszłe dopasowanie powinno
korzystać z wytrenowanych wag, inaczej niż obecne sprawdzenie konfiguracji.

## Dane i dalsza praca

Instrukcja pobrania MVTec AD i sprawdzenia loadera znajduje się w
[getting started](docs/getting-started.md). Dane, wytrenowane modele, cache
i wyniki pozostają poza Git. Ścieżki w YAML są liczone względem pliku konfiguracji.

Kolejność prac i zakres prostej aplikacji opisuje [roadmapa](docs/implementation-roadmap.md).
Wcześniejsze próby są opisane w [experiments](experiments/README.md).
