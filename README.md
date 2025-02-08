 == Model Architecture ==
- Hyperspectral Image Encoder

    - Uses a 3D CNN or a Transformer-based spectral encoder to extract spatial-spectral features.
    - Current direction:
        - 3D Convolutional Encoder: Extracts spatial and spectral features together.
        - Spectral Attention Transformer: Focuses on spectral dependencies and ignores spatial variations if only pixel spectra are used.
        - Pretrained HSI Feature Extractors (e.g., using contrastive learning on spectral data).

- Text Encoder

    - Use Transformer-based models (BERT, T5, CLIP-like text encoders) for textual feature extraction.
    - Alternative:
        - Traditional LSTMs or GRUs if computational resources are limited.
