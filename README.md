# IMDB Movie Review Sentiment Analysis

This project implements a sentiment analysis system for movie reviews using a pre-trained LSTM (Long Short-Term Memory) neural network model. The application can classify movie reviews as either positive or negative.

## Features

- Real-time sentiment analysis of movie reviews
- Web interface built with Streamlit
- Pre-trained LSTM model for accurate predictions
- Support for natural language input
- Confidence score for predictions

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── predictive_function.ipynb  # Jupyter notebook for prediction functions
├── simplernn.ipynb        # Jupyter notebook for RNN model development
├── embedding.ipynb        # Jupyter notebook for word embeddings
├── data/                  # Data directory
├── pickle/               # Directory for saved models
├── logs/                 # Training logs
└── image/                # Image assets
```

## Requirements

- Python 3.x
- TensorFlow
- Streamlit
- NumPy
- Other dependencies listed in pyproject.toml

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Enter a movie review in the text area

4. Click "Predict Sentiment" to get the classification result

## Model Details

The project uses a pre-trained LSTM model trained on the IMDB dataset. The model:
- Processes text through word embeddings
- Uses LSTM layers for sequence processing
- Provides binary classification (positive/negative)
- Includes confidence scores for predictions

## Development

The project includes several Jupyter notebooks for development and experimentation:
- `simplernn.ipynb`: Contains the RNN model development and training
- `embedding.ipynb`: Explores word embeddings
- `predictive_function.ipynb`: Implements prediction functions

## License

[Add your license information here]

## Contributing

[Add contribution guidelines if applicable]
