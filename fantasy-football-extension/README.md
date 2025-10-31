# Fantasy Football Deep Learning Extension

A comprehensive fantasy football prediction system using FT-Transformer deep learning models with Next Gen Stats data.

## 🎯 Overview

This project predicts fantasy football player performance (PPR scoring) using:
- **Next Gen Stats (NGS)** tracking data (receiving, rushing, passing metrics)
- **Game context** (Vegas lines, weather, injuries, snap counts)
- Optional engineered features (rolling averages, opponent strength, player trends)
- **FT-Transformer** neural network architecture for tabular data

## 📊 Features

- Position-specific models (QB, RB, WR, TE)
- Comprehensive feature engineering pipeline
- Time-series aware train/validation/test splits
- Hyperparameter optimization with Optuna
- Browser extension for real-time predictions

## 🏗️ Project Structure

```
fantasy-football-extension/
├── api/                          # Flask API for predictions
│   └── app.py
├── data/                         # Data storage
│   ├── raw/                      # Raw NFL data (parquet files)
│   └── processed/                # Cleaned and featured data
├── preprocessing/                # Data pipeline
│   └── clean_data.py            # Merge and clean raw data (no engineered features)
├── models/                       # Model training
│   ├── ft_transformer.py        # FT-Transformer architecture
│   └── train_position.py        # Training script
├── extension/                    # Browser extension
│   ├── manifest.json
│   ├── background.js
│   ├── content.js
│   └── styles.css
└── requirements.txt
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (optional, but recommended)
- 8GB+ RAM
- ~5GB disk space for data

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/MERC13/Deep-learning.git
cd fantasy-football-extension
```

2. **Create virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Data Collection

```bash
python data/collect_data.py
```

This downloads NFL data for seasons 2019-2024:
- Weekly player statistics
- Next Gen Stats (receiving, rushing, passing)
- Snap counts, injuries, Vegas lines

### Data Preprocessing

```bash
# Clean and merge data (feature engineering disabled)
python preprocessing/clean_data.py
```

### Model Training

```bash
python models/train_position.py
```

This trains models for all positions (QB, RB, WR, TE). Each model:
- Uses seasons 2019-2022 for training
- Season 2023 for validation
- Season 2024 for testing

**Training time:**
- GPU: ~10-15 minutes per position
- CPU: ~45-60 minutes per position

### API Server

```bash
python api/app.py
```

Server runs on `http://localhost:5000`

## 📈 Model Performance

Expected performance metrics (2024 season):

| Position | MAE (PPR) | RMSE | R² |
|----------|-----------|------|-----|
| QB | ~4-5 | ~6-7 | 0.60-0.70 |
| RB | ~5-6 | ~7-8 | 0.55-0.65 |
| WR | ~4-5 | ~6-7 | 0.50-0.60 |
| TE | ~3-4 | ~5-6 | 0.45-0.55 |

## 🎨 Browser Extension

### Installation

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select the `extension/` folder

### Usage

Navigate to fantasy football websites (ESPN, Yahoo, Sleeper, etc.) and the extension will automatically:
- Detect players on the page
- Display predicted fantasy points
- Show confidence intervals

## 🔧 Configuration

### Hyperparameters (default)

```python
{
    'd_token': 192,        # Token dimension
    'n_layers': 3,         # Number of transformer layers
    'n_heads': 8,          # Attention heads
    'dropout': 0.15,       # Dropout rate
    'lr': 1e-4,            # Learning rate
    'batch_size': 128      # Batch size
}
```

To enable hyperparameter optimization, uncomment in `train_position.py`:
```python
best_params = hyperparameter_optimization(data_dict, position, n_trials=20)
```

## 📊 Features Used

With feature engineering disabled, training and inference use columns available from the cleaned merges:

### Continuous features
- NGS Receiving: cushion, separation, intended air yards, YAC, catch % (plus counts like targets/receptions when present)
- NGS Rushing: efficiency, time to LOS, rush yards over expected, attempts/yards/TDs
- NGS Passing: time to throw, completed/intended air yards, air distance, attempts/yards/TDs/INTs, passer rating
- Workload: offensive snaps and snap percentage
- Weather (if present): temperature, wind

### Categorical features
- Teams: recent_team, opponent_team, depth_team
- Position
- Injury status fields: report_status, practice_status, report_primary_injury

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Code Style
```bash
black .
flake8 .
```

## 📝 Data Sources

- **nfl_data_py:** Official NFL data API
- **Next Gen Stats:** AWS-powered player tracking
- **Vegas Lines:** Betting odds and totals
- **Injury Reports:** Official NFL injury reports

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **nfl_data_py** for comprehensive NFL data access
- **FT-Transformer** paper by Yury Gorishniy et al.
- **rtdl** library for transformer implementations
- NFL Next Gen Stats powered by AWS

## 📧 Contact

Project Link: [https://github.com/MERC13/Deep-learning](https://github.com/MERC13/Deep-learning)

---

**Note:** This project is for educational and research purposes. Always verify predictions with your own analysis before making fantasy football decisions.
