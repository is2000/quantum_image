# Quantum Implicit Neural Representation (QIREN)

## Project Overview

This project explores a novel approach to **Implicit Neural Representation (INR)** using a **hybrid quantum-classical neural network**. Traditional INRs, while effective, often struggle to capture high-frequency details in signals, leading to blurry or indistinct outputs. Our research introduces the **Quantum Implicit Neural Representation (QIREN)**, which overcomes this limitation by leveraging the inherent power of quantum circuits.

The QIREN model uses a **data re-uploading** technique within its quantum circuit to exponentially enhance its ability to model the frequency spectrum of a signal. This allows the network to learn and represent fine-grained details more effectively than its classical counterparts. The project evaluates QIREN's performance on two key tasks: **image regression** and **image super-resolution**, demonstrating its superior capability in reconstructing and enhancing images.

---

## Getting Started

### Prerequisites

To run this project, you need to have the following installed on your system:

* **Python 3.8 or higher**
* **Jupyter Notebook** or **JupyterLab**

### Installation

Follow these steps to set up the project environment:

1.  **Clone the repository** from its source.
    ```bash
    git clone <repository_url>
    ```

2.  **Navigate into the project directory.**
    ```bash
    cd <project_directory>
    ```

3.  **Install the required Python packages.**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

The entire project workflow is contained within a single Jupyter Notebook. To run the experiments and see the results:

1.  **Start a Jupyter session** from your terminal.
    ```bash
    jupyter notebook
    ```

2.  **Open the notebook** named `Imagefittingpaperwithsuperresolutionfinal.ipynb`.

3.  **Run all the cells** in the notebook in order. The notebook will guide you through the process of setting up the model, training it for both image regression and super-resolution tasks, and displaying the comparative results with classical models.

---

## Results

Our experiments show that the QIREN architecture significantly outperforms a classical MLP with Random Fourier Features (RFF) encoding and ReLU activation. The superior performance of QIREN is particularly evident in the **Peak Signal-to-Noise Ratio (PSNR)**, a key metric for image quality.

| Task               | Model         | PSNR (dB) |
| :----------------- | :------------ | :-------- |
| Image Regression   | RFF + ReLU    | 25.49     |
| Image Regression   | Quantum Hybrid | **33.26** |
| Super-Resolution   | RFF + ReLU    | 20.07     |
| Super-Resolution   | Quantum Hybrid | **29.83** |

The results confirm that the QIREN model's ability to capture high-frequency details translates directly into higher-quality, more detailed reconstructed images.
