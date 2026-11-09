*Course*: Projekt przejściowy
*Project title*: Wykrywanie defektów produktów z wykorzystaniem analizy obrazów.

*Dataset(s)*:
-

*Goals*:
- Use CNN/LSTM to detect the object and it's anomalies, show exact defects if possible
- Attach simulated station parameters such as `station name/number`, `temperature` or other such measures
- Create a live gui with dahboard that will aggregate all stations data and anomaly detection events
- Make the `stations backed` send live signals to the model engine so the predictions are simulated real-time
- Find different datasets and test the solution on other items

## Current implementation decision

The selected method is our own PatchCore implementation in PyTorch, with a pretrained torchvision ResNet as the feature extractor. Anomalib and EfficientAD are reviewed alternatives, not target implementation dependencies or additional model milestones. The original CNN/LSTM ideas above describe the initial proposal; LSTM and learned sensor/image fusion are outside the initial scope. See the implementation roadmap for the revised order of work.
