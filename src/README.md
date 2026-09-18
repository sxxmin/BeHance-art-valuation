# src

## Predictor label updates

Some predictor labels have been updated for clarity and consistency in the revised manuscript. The table below maps earlier labels to their revised forms. Earlier labels may remain in code or data files; this table provides a reference for matching them to the manuscript.

| Earlier label | Revised manuscript label |
| --- | --- |
| Complexity-entropy (C-H) values per image | Complexity–entropy (C–H) measures |
| Hue proportion (bin 0-7) (Mean, median) per image (dummy) | Binned hue mean and median (bins 0–7) |
| Hue (Standard deviation) per image | Hue standard deviation |
| Saturation and value statistics (Mean, median, standard deviation) per image | Saturation and value statistics |
| Red, green, and blue (RGB) channel statistics (Mean, median, standard deviation) per image | RGB-channel statistics |
| Latent PCA embeddings (dim 1-200) per image | PCA-reduced ResNet-18 layer representations |
| Assigned creative field count | Assigned creative-field count |
| List of unique creative fields | Creative-field indicators |
| Neighboring distance across project-wise C-H values | Neighboring-project C–H distance |
| Neighboring distance across project-wise HSV channels | Neighboring-project HSV distance |
| Neighboring distance across project-wise RGB channels | Neighboring-project RGB distance |
| Neighboring similarity across project-wise latent PCA embeddings | Neighboring-project latent-PCA similarity |
| Platform membership join year (dummy) | Platform membership year |
| Residency (countries) | Residency |
| Topic entropy (diversity of unique creative field combinations across an artist's career timeline) | Topic entropy |
| Curation status of the last artwork (Y/N) | Preceding-project curation status (Y/N) |
| Data acquisition-to-project publication interval (days) | Interval from project publication to data acquisition (days) |

Minor spelling and formatting changes include `Inter project publication period` → `Inter-project publication period`, and standardized spacing in creative-field names, such as `ArtDirection` → `Art Direction` and `GraphicDesign` → `Graphic Design`.

In the manuscript text, "curation status of the preceding project" refers to the predictor listed as "Preceding-project curation status." Shortened labels used in figures and supplementary tables refer to the corresponding predictors listed above.

These naming changes do not themselves alter predictor values or analytical results.

[← Back to main README](../README.md)
