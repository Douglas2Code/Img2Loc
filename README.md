# Img2Loc: Revisiting Image Geolocalization using Multi-modality Foundation Models and Image-based Retrieval-Augmented Generation

Code base for Img2Loc paper presented on SIGIR 2024.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

## Installation

Instructions on how to install and set up the project.

```bash
# Clone the repository
git clone git@github.com:Douglas2Code/Img2Loc.git

# Change to the project directory
cd Img2Loc

# Create a conda environment
conda create -n img2loc python=3.10 -y

# Activate the conda environment
conda activate img2loc

# Install faiss databse following this guide
https://github.com/facebookresearch/faiss/blob/main/INSTALL.md

# Install the project dependencies
pip install -r requirements.txt
```

## Usage

Run the streamlip application

```python
streamlit run app.py --browser.gatherUsageStats false
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Zhongliang Zhou: zzldouglas97@gmail.com
Jielu Zhang: jz20582@uga.edu
