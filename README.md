# Img2Loc: Revisiting Image Geolocalization using Multi-modality Foundation Models and Image-based Retrieval-Augmented Generation

Code for Img2Loc paper presented on SIGIR 2024.

![Banner](./static/figure3.jpg)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

## Installation

Instructions on how to install and set up the project. If you needs help to access the generated embeddings, please contact us directly.

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

# Download MP16 dataset
http://www.multimediaeval.org/mediaeval2016/placing/

# Generate embeddings using CLIP model
https://github.com/openai/CLIP

# Generate a vector database using FAISS
https://github.com/facebookresearch/faiss/wiki/Getting-started#in-python-1

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
