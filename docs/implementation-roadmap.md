# Roadmapa implementacji

Aktualizacja: 11 listopada 2026 r. Zakres: lokalna aplikacja do analizy obrazów, porównania PatchCore i PaDiM oraz pobierania wyników. Obie metody będą korzystać z Anomalib.

## Architektura docelowa

```text
wybór obrazów i modelu w Streamlit
    -> wspólna funkcja analizy z zapisanym modelem Anomalib
    -> wyniki, mapy i tabela porównawcza
    -> pobranie CSV/JSON oraz archiwum wizualizacji
```

Dopasowanie i kalibracja są oddzielone od analizy nowych zdjęć. Prosty moduł obliczeniowy jest wywoływany bezpośrednio z aplikacji, a wyniki są przechowywane w pamięci sesji i lokalnych plikach. Nie planujemy osobnego API, bazy danych, kolejki, Spark/Kafka, odtwarzania zdarzeń ani symulatora stanowisk. Infrastruktura projektu Big Data nie jest częścią tego projektu.

## Stan i kolejność

| Etap | Zakres | Stan |
| --- | --- | --- |
| 0 | Zależności, lockfile i konfiguracje obu modeli | Gotowe do sprawdzenia bez danych i wag |
| 1 | Loader MVTec AD, podziały i podgląd masek | Dostępne |
| 2 | Dopasowanie, zapis i odczyt modeli Anomalib | Do wykonania |
| 3 | Kalibracja i wspólna analiza obrazów | Do wykonania |
| 4 | Porównanie jakości, czasu i pamięci | Do wykonania |
| 5 | Aplikacja Streamlit, wyniki i eksport | Do wykonania |

W repozytorium zachowano wcześniejszy własny PatchCore. Jest eksperymentem dostępnym pod `defect native ...`, a nie gotową integracją nowych modeli. Dotychczasowe sprawdzenia tego prototypu nie są wynikami porównania PatchCore/PaDiM z Anomalib.

## 0–1. Środowisko i dane

- Korzystać z `uv sync --locked` i Pythona 3.12. Anomalib oraz Streamlit należą do głównych zależności.
- Wczytać dwa pliki `configs/mvtec-bottle-{patchcore,padim}.yaml` przez `load_model_config`.
- Tworzyć modele przez `build_model`, z jawnym wyborem `pretrained`. Sprawdzenie konfiguracji używa `False`; przyszłe dopasowanie wymaga wag wytrenowanego ekstraktora.
- Obecny wspólny preprocessing skaluje do 256 × 256 bez cropu i normalizuje według ImageNet. Loader oddaje obrazy RGB w `[0, 1]`; nie normalizować ich podwójnie.
- Zachować manifest i rozdzielenie danych do dopasowania, walidacji, kalibracji i końcowego testu. Obrazy testowe i maski nie służą do budowy modelu ani wyboru progu.

Instrukcja działających komend: [getting started](getting-started.md). Sprawdzenie konstrukcji modeli nie wymaga danych, nie uczy modeli i nie ocenia jakości.

## 2. Dopasowanie i artefakty Anomalib

- Podłączyć oba modele do tych samych ustalonych podziałów danych.
- Użyć pretrained ResNet-18, zapisując warstwy, parametry właściwe metodom, seed i wersję biblioteki.
- Przygotować wywołanie dopasowania oraz zapis i odczyt kompletnego stanu modelu.
- Zapisywać tożsamość metody, implementacji, preprocessingu i podziału danych. Nie wczytywać bez sprawdzenia starszych natywnych checkpointów jako modeli Anomalib.
- Potwierdzić zgodność predykcji po odczycie w nowym procesie, bez ponownego dopasowania i pobierania wag.

## 3. Kalibracja i analiza

- Wyznaczyć dla każdego modelu osobny próg obrazu i mapy na wydzielonych poprawnych obrazach. Domyślny postprocessor jest obecnie wyłączony; samo utworzenie modelu nie zapewnia skalibrowanej decyzji.
- Przygotować prostą funkcję analizy przyjmującą model i listę obrazów, zwracającą surowy wynik, decyzję, mapę, czas oraz identyfikator konfiguracji.
- Obsłużyć pojedyncze zdjęcia i małą partię obrazów. Zachować oryginalne rozmiary przy wizualizacji i stałą skalę kolorów.
- Ładować zapisany model ponownie tylko przy zmianie jego wyboru, a nie trenować go przy każdym odświeżeniu aplikacji.

## 4. Porównanie metod

- Użyć tych samych próbek testowych, ekstraktora, rozdzielczości i protokołu pomiarowego. Zapisać różnice w warstwach i redukcji wymiaru cech.
- Osobno ocenić obrazy i lokalizację: AUROC/AP, precyzja i czułość po progowaniu, pikselowa AP i AU-PRO z określonym zakresem.
- Zmierzyć koszt dopasowania, wielkość modelu, pamięć oraz czas predykcji na tym samym sprzęcie, po rozgrzaniu i przy tej samej wielkości partii.
- Rozdzielić wyniki benchmarku z etykietami od analizy zdjęć użytkownika bez etykiet. Dla tych drugich nie pokazywać wyliczonej skuteczności bez danych referencyjnych.
- Pokazać wyniki dla kategorii, przykłady błędów i ograniczenia kalibracji. Dodatkowe konfiguracje lub VisA są rozszerzeniem po podstawowym porównaniu.

## 5. Prosta aplikacja i eksport

- Streamlit: wybór plików, wybór jednej lub obu metod i przycisk uruchomienia analizy.
- Widok zdjęcia, mapy i nakładki oraz tabela wyników obu modeli.
- Osobny widok zestawienia benchmarkowego: jakość, czas i zasoby wraz z konfiguracją badania.
- Pobranie tabeli CSV/JSON i paczki ZIP z wizualizacjami. Eksport powinien zawierać nazwy plików, metody, wyniki, progi i konfigurację, aby dało się interpretować go poza aplikacją.
- Jasne komunikaty dla braku modelu, niepoprawnego obrazu i pustego zestawu danych. Brak błędu nie może być zastępowany decyzją o poprawnym produkcie.

## Kryterium zakończenia

Kolega z zespołu uruchamia aplikację według instrukcji, wybiera obrazy, uzyskuje wyniki PatchCore i PaDiM, przegląda porównanie oraz pobiera pliki. Raport opisuje rzeczywiście wykonane badania i ograniczenia, a nie wyniki przepisane z publikacji.
