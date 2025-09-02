# Real LRU Models for Audio Distortion Circuits

This repository accompanies the paper:

> **Antialiased Black-Box Modeling of Audio Distortion Circuits using Real Linear Recurrent Units**  
> Fabián Esqueda, Shogo Murai — DAFx25 (2025)  
> [Read the paper](https://dafx25.dii.univpm.it/wp-content/uploads/2025/07/DAFx25_paper_61.pdf)

It provides a PyTorch implementation of the proposed LRU-based model for black-box modeling of nonlinear audio distortion circuits, such as diode clippers and overdrive pedals.

---

## Overview

For quick reference, a minimal training script (`train_dummy.py`) is included.  
It shows how to instantiate and train the model (`model.py`) on a toy dataset of white-noise sequences processed through a `tanh` nonlinearity.  
This example also illustrates the expected input/output tensor format: `(batch_size, sequence_length, 1)`.

- Uses [`mambapy`](https://github.com/alxndrTL/mamba.py) for the efficient prefix-scan (`pscan`) operation.

---

## Sound examples:

https://drive.google.com/drive/folders/1eRGrj4K2HJxhHrso3v5dbewsRUTHJo-9?usp=drive_link

---

## 📦 Installation

Clone this repo and install dependencies:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install torch mambapy
