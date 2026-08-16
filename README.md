# **`README.md`**

# Regime-Gated Residual Mixture-of-Experts for Cross-Sectional Volatility Forecasting

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Source](https://img.shields.io/badge/Source-arXiv%20Preprint-B31B1B)](https://arxiv.org/abs/2608.12251)
[![Year](https://img.shields.io/badge/Year-2026-purple)](https://github.com/chirindaopensource/regime_gated_residual_mixture_of_experts_for_cross_sectional_volatility_forecasting)
[![Status](https://img.shields.io/badge/Status-Independent%20Implementation-brightgreen)](https://github.com/chirindaopensource/regime_gated_residual_mixture_of_experts_for_cross_sectional_volatility_forecasting)
[![Discipline: Financial Econometrics](https://img.shields.io/badge/Discipline-Financial%20Econometrics-00529B)](https://github.com/chirindaopensource/regime_gated_residual_mixture_of_experts_for_cross_sectional_volatility_forecasting)
[![Discipline: Deep Learning](https://img.shields.io/badge/Discipline-Deep%20Learning-00529B)](https://github.com/chirindaopensource/regime_gated_residual_mixture_of_experts_for_cross_sectional_volatility_forecasting)
[![Discipline: Time-Series Analysis](https://img.shields.io/badge/Discipline-Time--Series%20Analysis-00529B)](https://github.com/chirindaopensource/regime_gated_residual_mixture_of_experts_for_cross_sectional_volatility_forecasting)
[![Discipline: Risk Management](https://img.shields.io/badge/Discipline-Risk%20Management-00529B)](https://github.com/chirindaopensource/regime_gated_residual_mixture_of_experts_for_cross_sectional_volatility_forecasting)
[![Method: Mixture-of-Experts](https://img.shields.io/badge/Method-Mixture--of--Experts-orange)](https://github.com/chirindaopensource/regime_gated_residual_mixture_of_experts_for_cross_sectional_volatility_forecasting)
[![Method: Residual Learning](https://img.shields.io/badge/Method-Residual%20Learning-orange)](https://github.com/chirindaopensource/regime_gated_residual_mixture_of_experts_for_cross_sectional_volatility_forecasting)
[![Method: Walk-Forward Validation](https://img.shields.io/badge/Method-Walk--Forward%20Validation-orange)](https://github.com/chirindaopensource/regime_gated_residual_mixture_of_experts_for_cross_sectional_volatility_forecasting)
[![Method: Diebold-Mariano](https://img.shields.io/badge/Method-Diebold--Mariano%20Testing-orange)](https://github.com/chirindaopensource/regime_gated_residual_mixture_of_experts_for_cross_sectional_volatility_forecasting)
[![Method: Kupiec VaR](https://img.shields.io/badge/Method-Kupiec%20VaR%20Calibration-orange)](https://github.com/chirindaopensource/regime_gated_residual_mixture_of_experts_for_cross_sectional_volatility_forecasting)
[![Data: Yahoo Finance US](https://img.shields.io/badge/Data-Yahoo%20Finance%20US%20Equities-lightgrey)](https://finance.yahoo.com/)
[![Data: Yahoo Finance JP](https://img.shields.io/badge/Data-Yahoo%20Finance%20TSE%20Prime-lightgrey)](https://finance.yahoo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%238CAAE6.svg?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-brightgreen)](https://github.com/chirindaopensource/regime_gated_residual_mixture_of_experts_for_cross_sectional_volatility_forecasting)

**Repository:** `https://github.com/chirindaopensource/regime_gated_residual_mixture_of_experts_for_cross_sectional_volatility_forecasting`

**Owner:** 2026 Craig Chirinda (Open Source Projects)

## Table of Contents
- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: orchestrate_research_pipeline](#key-callable-orchestrate_research_pipeline)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project is an **independent, professional-grade implementation** of the theoretical models, architectural designs, and econometric evaluation protocols from the arXiv preprint titled **"Regime-Gated Residual Mixture-of-Experts for Cross-Sectional Volatility Forecasting"** by the authors:
*   **Junyi Ye**
*   **Gargi Vijay Borde**

Note: *The preprint (arXiv:2608.12251, e-journal submission date 12 August 2026) is referenced throughout this repository as an **arXiv Preprint**; it is not a peer-reviewed journal article, and this implementation does not claim any journal affiliation.*

This repository provides a complete, end-to-end computational framework that rigorously answers a critical architectural question in financial machine learning: *where should nonstationary regime information enter a neural forecasting model?* By holding information, model capacity, hyperparameter tuning, and random seeds strictly constant, this pipeline demonstrates that appending regime variables directly to forecasting inputs degrades both predictive performance and training stability. Conversely, restricting regime information to a soft-routing gate that modulates zero-initialized residual corrections—the **RG-ResMoE** architecture—delivers superior accuracy, eliminates training collapse, and significantly improves Value-at-Risk (VaR) calibration during high-stress market conditions.

## Theoretical Background

The implemented framework models cross-sectional equity volatility using a hybrid econometric and deep learning approach.

**1. The Forecasting Target:** For stock $i$ on day $t$, the target is the annualized five-day forward realized volatility, computed from daily log returns $r_{i,t} = \ln(P_{i,t}/P_{i,t-1})$:
$$y_{i,t} = \sqrt{252} \, \text{Std}(r_{i,t+1}, \dots, r_{i,t+5})$$

**2. The Shared MLP Block:** All neural architectures are constructed from a common two-hidden-layer Multi-Layer Perceptron (MLP) block with GELU activations ($\phi$) and dropout ($p=0.1$):
$$h_1 = \phi(W_1 q + b_1), \quad h_2 = \phi(W_2 h_1 + b_2), \quad \text{Block}(q; \theta) = W_3 h_2 + b_3$$

**3. The RG-ResMoE Architecture:** The proposed model decomposes the forecast into a frozen base prediction and a soft-gated residual correction. The base network and $K=4$ residual experts consume strictly stock-level features $x \in \mathbb{R}^{16}$, while the gating network consumes both features and regime state variables $u = (x, z) \in \mathbb{R}^{18}$:
$$\hat{y}_{\text{base}} = \text{Block}(x; \theta_b)$$
$$r_k(x) = \text{Block}(x; \theta_k)$$
$$\pi = \text{softmax}(g(u))$$
$$\hat{y} = \hat{y}_{\text{base}} + \sum_{k=1}^K \pi_k r_k(x)$$
Crucially, the final layers of the residual experts are **zero-initialized**, ensuring that $r_k(x) = 0$ at initialization and the model begins by exactly reproducing the frozen base forecast.

**4. The Regularized Objective Function:** Stage 2 training optimizes the experts and the gate jointly using a composite loss function that penalizes the aggregate residual magnitude and discourages routing collapse:
$$\mathcal{L} = \text{MSE}(y, \hat{y}) + \alpha \overline{\left(\sum_{k=1}^K \pi_k r_k(x)\right)^2} + \lambda_{\text{LB}} \sum_{k=1}^K \left(\bar{\pi}_k - \frac{1}{K}\right)^2$$

**5. Statistical Inference and Calibration:** Model superiority is evaluated using paired Diebold-Mariano tests on seed-averaged daily Information Coefficient (IC) differences, utilizing Newey-West standard errors with 4 lags to correct for the $\text{MA}(4)$ dependence induced by overlapping targets. Risk calibration is assessed via Kupiec's unconditional coverage likelihood-ratio test on Student-$t$ VaR thresholds.

## Features

- **Information-Matched Architectural Ablations:** Rigorous isolation of the integration pathway. Compares input concatenation (`MLP-L(+z)`) against gate-based routing (`RG-ResMoE`) while holding parameter capacity ($\approx 2,800$ parameters) and random seeds strictly constant.
- **Two-Stage Residual Training Engine:** Implements full-batch Adam optimization with a frozen base network and zero-initialized residual branches, completely eliminating the training collapse endemic to standard Mixture-of-Experts models.
- **Robust Hyperparameter Selection:** A 3-filter validation protocol that rejects boundary configurations, enforces statistical IC dominance over runner-ups, and guarantees MSE robustness without ever touching the test split.
- **Comprehensive Econometric Baselines:** Includes pooled Heterogeneous Autoregressive (HAR) models, per-stock constrained GARCH(1,1) Maximum Likelihood Estimation, 20-day persistence, and L2-regularized Ridge regression.
- **Stress-Condition Slicing:** Automated evaluation of model performance during the COVID-19 crash, the 2022 bear market, top-decile volatility days, and regime-flip windows.
- **Cross-Market Replication:** A fully parameterized pipeline that seamlessly replicates the U.S. study on an independent Japanese TSE Prime panel to verify global generalization.
- **Cryptographic Reproducibility:** Generates a deterministic SHA-256 manifest linking data hashes, configuration versions, seed registries, and final results tables.

## Methodology Implemented

The proposed research approach can be distilled into the following principal steps:

1. **Configuration and Schema Validation:** Strict auditing of the `config.yaml` file, enforcing mathematical bounds, walk-forward geometric identities, and exact parameter capacity matching.
2. **Data Acquisition and Cleansing:** Downloading Yahoo Finance OHLCV bars, computing split/dividend-adjusted log returns, and removing confirmed corporate-action artifacts.
3. **Feature and Regime Engineering:** Constructing 16 stock-specific features (realized volatility, cumulative returns, RSI, lagged returns) and 2 regime variables (systematic market volatility and 120-day rolling OLS idiosyncratic volatility).
4. **Walk-Forward Tensor Assembly:** Generating 30 non-overlapping quarterly test windows, fitting $z$-score normalization moments strictly on the 85% training splits, and assembling PyTorch tensors.
5. **Model Construction:** Instantiating the shared MLP blocks, capacity-matched baselines, standard MoE, RG-ResMoE, and hard-routing variants (learned top-1, volatility quantiles, GICS sectors, market $\times$ idio split).
6. **Hyperparameter Tuning:** Executing 3-seed sweeps across learning rates and regularizers ($\alpha, \lambda_{\text{LB}}$), applying the 3-filter robustness protocol.
7. **Full-Batch Training:** Executing 30-seed training runs with early stopping and best-checkpoint restoration.
8. **Evaluation and Inference:** Computing IC, ICIR, RMSE, $R^2$, and QLIKE. Executing 4-lag Newey-West Diebold-Mariano tests and Student-$t$ Kupiec VaR calibration.
9. **Cross-Market Replication:** Re-executing the entire pipeline on the Japanese TSE Prime panel.
10. **Robustness Auditing:** Generating stress-slice metrics, evaluating gate softness, and compiling the final Markdown audit report with a `FAITHFUL` reproduction verdict.

The illustration below shows the Inputs-Processes-Outputs of the proposed research methodology:
<div align="center">
  <img src="https://github.com/chirindaopensource/regime_gated_residual_mixture_of_experts_for_cross_sectional_volatility_forecasting/blob/main/regime_gated_residual_mixture_of_experts_for_cross_sectional_volatility_forecasting_ipo_main_1.jpg" alt="Pipeline Architecture" width="100%">
</div>

## Core Components (Notebook Structure)

*Note: All orchestrator callables and their constituent helper functions are contained within a singular, comprehensive Jupyter Notebook (`regime_gated_residual_mixture_of_experts_for_cross_sectional_volatility_forecasting_draft.ipynb`).*

The notebook is structured as a logical sequence of 18 distinct tasks, a consolidated `# Import Essential Modules` cell, and a top-level orchestrator. Key sections include:
- **Task 1:** Validate Configuration Quality and Input Schema Contracts
- **Task 2:** Reproducible Python Environment and Raw Data Acquisition
- **Task 3:** Cleanse Raw Price Data and Construct Aligned Return Panels
- **Task 4:** Construct Stock Features, Regime Variables, and Forward Targets
- **Task 5:** Implement Walk-Forward Splits and Training-Only Normalization
- **Task 6:** Implement the Shared MLP Block and the Gate Network
- **Task 7:** Implement Capacity-Matched MLP Baselines and Standard MoE
- **Task 8:** Implement RG-ResMoE Family and Hard-Routing Variants
- **Task 9:** Implement Classical Baselines
- **Task 10:** Implement the Full-Batch Training Engine
- **Task 11:** Hyperparameter Selection Protocol
- **Task 12:** Compute Forecast Evaluation Metrics
- **Task 13:** Statistical Inference with Diebold–Mariano and Newey–West
- **Task 14:** Value-at-Risk Calibration and Kupiec Testing
- **Task 15:** Replicate the Japanese Cross-Market Study
- **Task 16:** Create the End-to-End Orchestrator Callable
- **Task 17:** Robustness Analyses and Final Verification
- **Task 18:** Top-Level End-to-End Research Pipeline Orchestrator

## Key Callable: `orchestrate_research_pipeline`

The project is designed around a single, top-level user-facing interface function:

- **`orchestrate_research_pipeline`:** This apex orchestrator executes the complete research protocol in dependency order. A single call validates the configuration, builds the model families, runs the per-market data pipelines, executes the 3-seed hyperparameter sweeps and 30-seed training runs, evaluates all econometric metrics, performs the Japanese cross-market replication, conducts the stress-condition robustness analysis, and persists all artifacts (tensors, checkpoints, figures, tables, and cryptographic manifests) to disk.

```python
def orchestrate_research_pipeline(
    config: Dict[str, Any],
    raw_root: Path,
    output_root: Path,
    seed_list: List[int],
    markets: List[str] = ("US", "JP"),
) -> Dict[str, Any]:
```

## Prerequisites

- Python 3.11+
- Core Python dependencies: `torch`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `pyyaml`.
- Optional for automated data acquisition: `yfinance`, `pandas_market_calendars`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/regime_gated_residual_mixture_of_experts_for_cross_sectional_volatility_forecasting.git
    cd regime_gated_residual_mixture_of_experts_for_cross_sectional_volatility_forecasting
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Input Data Structure

The pipeline requires raw daily price histories, universe metadata, and a static YAML configuration file (`config.yaml`) saved in the working directory:

- **Raw Price CSVs**: One file per ticker (e.g., `AAPL.csv`, `7203.T.csv`) containing standard Yahoo Finance columns: `date`, `open`, `high`, `low`, `close`, `adj_close`, `volume`. Only `adj_close` is used for return construction.
- **Universe Metadata**: A CSV file mapping tickers to `gics_sector`, `sp1500_member_current`, `tse_prime_member`, and `market_cap_bucket` to support hard-routing variants and universe filtering.
- **`config.yaml`**: The serializable master configuration defining all hyperparameters, walk-forward geometries, architectural widths, regularizer grids, and evaluation thresholds. It acts as the single source of truth for the entire pipeline.

## Usage

Here is the granular, step-by-step guide to executing the end-to-end pipeline for **"Regime-Gated Residual Mixture-of-Experts for Cross-Sectional Volatility Forecasting"**. This example demonstrates how to load the study configuration from a YAML file, set up the required directory structures, and execute the full research pipeline using the `orchestrate_research_pipeline` orchestrator.

*Note: This example assumes that all the callables defined in this conversation (including `orchestrate_research_pipeline`, `validate_config_and_schemas`, etc.) are available in the current namespace, such as within a single Jupyter notebook.*

### **Step 1: Loading the Configuration (`config.yaml`)**

The study relies on a deterministic configuration file (`config.yaml`) that defines all hyperparameters, data splits, architectural details, and evaluation metrics. We assume this file exists in the working directory.

**Methodology:**
1.  **File I/O:** Open `config.yaml` in read mode.
2.  **Parsing:** Use `yaml.safe_load` to convert the YAML structure into a nested Python dictionary.
3.  **Validation:** Catch file existence errors and parsing errors to ensure the pipeline does not proceed with an invalid configuration.

```python
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

def load_study_configuration(filepath: str = "config.yaml") -> Dict[str, Any]:
    """
    Loads the study configuration parameters from a YAML file into a Python dictionary.

    Purpose:
        To ingest the deterministic hyperparameters, data split definitions, architectural
        specifications, and evaluation metrics defined in the external configuration file.
        This ensures reproducibility by separating code from configuration.

    Inputs:
        filepath (str): The relative or absolute path to the YAML configuration file.
                        Default is "config.yaml".

    Processes:
        1.  File Access: Attempts to open the specified file in read mode with UTF-8 encoding.
        2.  Parsing: Uses PyYAML's safe_load to parse the YAML structure into a dictionary.
        3.  Validation: Catches FileNotFoundError and yaml.YAMLError, raising informative
            exceptions to halt execution if the configuration cannot be loaded.

    Outputs:
        Dict[str, Any]: A nested dictionary containing the complete study configuration.

    Raises:
        TypeError: If filepath is not a string.
        FileNotFoundError: If the specified YAML file does not exist.
        yaml.YAMLError: If the file contains invalid YAML syntax.
    """
    # Validate that the filepath is a string
    if not isinstance(filepath, str):
        # Raise a TypeError if the filepath is invalid
        raise TypeError(f"filepath must be a string, got {type(filepath).__name__}.")

    # Convert the string filepath to a Path object for robust handling
    config_path: Path = Path(filepath)

    # Verify that the configuration file exists
    if not config_path.is_file():
        # Raise a FileNotFoundError if the file is missing
        raise FileNotFoundError(f"Configuration file not found at: {config_path.absolute()}")

    try:
        # Open the configuration file in read mode with UTF-8 encoding
        with config_path.open("r", encoding="utf-8") as file:
            # Parse the YAML content safely into a Python dictionary
            config: Dict[str, Any] = yaml.safe_load(file)

        # Log success to the console
        print(f"\nSuccessfully loaded configuration from {config_path.name}")

        # Return the parsed configuration dictionary
        return config

    except yaml.YAMLError as exc:
        # Catch YAML parsing errors and raise an informative exception
        raise ValueError(f"Error parsing YAML file {config_path.name}: {exc}") from exc

# Load the configuration
# Note: Ensure 'config.yaml' is in your working directory with the content provided previously.
study_config: Dict[str, Any] = load_study_configuration("config.yaml")
```

### **Step 2: Executing the Pipeline (`orchestrate_research_pipeline`)**

With the configuration (`study_config`) in memory, we invoke the top-level orchestrator. This function manages the entire lifecycle: configuration validation, data acquisition (if raw data is missing), panel alignment, feature engineering, walk-forward splitting, model training (including the two-stage RG-ResMoE protocol), hyperparameter selection, evaluation, statistical testing, and robustness analysis.

**Methodology:**
1.  **Directory Setup:** Define the root directories for raw data (`raw_root`) and pipeline artifacts (`output_root`).
2.  **Parameter Extraction:** Extract the required `seed_list` and `markets` from the loaded configuration to ensure consistency.
3.  **Function Call:** Pass the configuration, directories, seeds, and markets to `orchestrate_research_pipeline`.
4.  **Output Handling:** The function returns a dictionary containing all experimental artifacts, including the final `StudyReport`.

```python
# ==============================================================================
# Execution of the End-to-End Study Pipeline
# ==============================================================================

if __name__ == "__main__":
    # Ensure we have a valid configuration before running
    if study_config:
        print("Initiating RG-ResMoE Study Pipeline...")

        # 1. Define Root Directories
        # Define the directory where raw Yahoo Finance CSVs will be stored or read from
        raw_data_root: Path = Path("./data/raw")
        # Define the directory where all pipeline artifacts (panels, models, metrics) will be saved
        pipeline_output_root: Path = Path("./output")

        # 2. Extract Execution Parameters from Configuration
        # Extract the deterministic seed registry from the reproducibility section
        seed_registry: List[int] = study_config.get("reproducibility_and_audit", {}).get(
            "seed_values", list(range(30))
        )
        # Define the target markets for the study (U.S. main study and Japanese replication)
        target_markets: List[str] = ["US", "JP"]

        try:
            # 3. Execute the Pipeline
            # Invoke the master orchestrator to run all 18 tasks
            # Note: This process is computationally intensive and may take significant time
            # depending on hardware and the presence of cached checkpoints.
            study_artifacts: Dict[str, Any] = orchestrate_research_pipeline(
                config=study_config,
                raw_root=raw_data_root,
                output_root=pipeline_output_root,
                seed_list=seed_registry,
                markets=target_markets
            )

            # ==============================================================================
            # Inspecting the Outputs
            # ==============================================================================

            print("\n" + "="*80)
            print("STUDY EXECUTION COMPLETE")
            print("="*80)

            # Extract the final StudyReport from the robustness stage artifacts
            # The robustness stage finalizes the audit report and decision log
            robustness_report = study_artifacts.get("task_17_robustness")

            if robustness_report:
                # 1. Accessing the Stress-Condition Results
                print("\n[Stress-Condition Analysis]")
                print("Performance of RG-ResMoE vs. MLP-L during market stress:")
                print(robustness_report.stress_table.to_string(index=False))

                # 2. Accessing the Robustness Results
                print("\n[Robustness Analysis]")
                print("Evaluation of hard-routing variants and ablations:")
                # Display a subset of columns for readability
                display_cols = ["variant", "ic", "collapsed", "p_value"]
                print(robustness_report.robustness_table[display_cols].to_string(index=False))

                # 3. Locating the Audit Report and Decision Log
                print("\n[Artifact Locations]")
                print(f"Final Audit Report: {robustness_report.audit_report_path}")
                print(f"Decision Log:       {robustness_report.decision_log_path}")

            # Extract the Japanese replication report
            jp_report = study_artifacts.get("task_15_japan")
            if jp_report:
                # 4. Accessing the Japanese Replication Results
                print("\n[Japanese Cross-Market Replication]")
                print("Performance on the TSE Prime panel:")
                display_cols_jp = ["model", "mean_ic", "mean_rmse", "mean_qlike", "collapsed_seeds"]
                print(jp_report.results_table[display_cols_jp].to_string(index=False))

        except Exception as e:
            # Catch and display any errors that occurred during pipeline execution
            print(f"\nPipeline execution failed: {e}")
            raise

    else:
        print("Error: Missing configuration. Cannot proceed.")
```

### **Summary of the Execution Flow**

1.  **Configuration Ingestion:** The pipeline begins by loading the deterministic `config.yaml` file, which dictates all hyperparameters, architectural choices, and evaluation rules.
2.  **Validation & Setup (Task 1 & 6-9):** The orchestrator validates the configuration structure, verifies mathematical constraints, and instantiates all model factories (Shared Blocks, MLP Baselines, Standard MoE, RG-ResMoE variants, and Classical Baselines), ensuring capacity alignment.
3.  **Data Acquisition & Preprocessing (Tasks 2-5):** For each market (US, JP), the pipeline downloads raw data (if absent), cleanses prices, aligns returns to a common calendar, computes 16 features and 2 regime variables, generates the 5-day forward target, and creates 30 normalized walk-forward tensor bundles.
4.  **U.S. Market Execution (Tasks 10-14):** The pipeline performs 3-seed hyperparameter sweeps on the first window, executes 30-seed full-batch training across all windows (using the two-stage protocol for RG-ResMoE), evaluates out-of-sample metrics (IC, RMSE, QLIKE), runs Diebold-Mariano statistical tests, and calibrates Student-t VaR thresholds.
5.  **Japanese Replication (Task 15):** The pipeline repeats the tuning, training, and evaluation process on the TSE Prime panel, verifying that the qualitative stability and accuracy signatures hold across markets.
6.  **Robustness & Audit (Tasks 16-17):** Finally, the pipeline computes stress-slice metrics, evaluates gate softness, asserts robustness constraints (e.g., soft > hard routing), finalizes the decision log, and generates a comprehensive Markdown audit report with a `FAITHFUL` reproduction verdict.

## Output Structure

The pipeline returns a comprehensive artifacts dictionary and writes a deterministic directory tree under `output_root`:

- **`raw/`** — The downloaded Yahoo Finance CSVs and cryptographic manifests.
- **`panels/`** — The cleansed, aligned, and feature-engineered Parquet panels.
- **`splits/`** — The 30 normalized walk-forward tensor bundles and row-order metadata.
- **`models/`** — Serialized PyTorch model architectures.
- **`checkpoints/`** — Resumable intermediate training states and stage-completion `.done` markers.
- **`forecasts/`** — Out-of-sample predictions aligned by ticker and date.
- **`metrics/`** — Evaluation tables, Diebold-Mariano JSON reports, and Kupiec VaR summaries.
- **`figures/`** — The generated plots, including the per-seed QLIKE stability scatter plot.
- **`logs/`** — The `decision_log.json` (documenting all parameter resolutions) and `reproducibility_manifest.json` (linking data hashes, seeds, and results).

## Project Structure

```
regime_gated_residual_mixture_of_experts_for_cross_sectional_volatility_forecasting/
│
├── regime_gated_residual_mixture_of_experts_for_cross_sectional_volatility_forecasting_draft.ipynb
├── config.yaml
├── requirements.txt
├── LICENSE
├── README.md
│
└── output/
    ├── raw/
    ├── panels/
    ├── splits/
    ├── models/
    ├── checkpoints/
    ├── forecasts/
    ├── metrics/
    ├── figures/
    └── logs/
        ├── decision_log.json
        └── reproducibility_manifest.json
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Researchers and risk managers can modify the framework without altering a single line of Python code:

- **Walk-Forward Geometry:** Adjust `development_length_trading_days`, `test_length_trading_days`, or `number_of_windows` to evaluate performance over different horizons.
- **Regularization Grids:** Expand or refine the `alpha_candidate_grid` and `lambda_LB_candidate_grid` to test alternative shrinkage and load-balancing strengths.
- **Architectural Widths:** Modify `base_network_width` or `expert_width` to scale the capacity of the RG-ResMoE model.
- **Hard-Routing Rules:** Redefine the `hard_routing_gics_sectors.mapping` to test alternative industry clustering strategies.
- **Risk Calibration:** Adjust the `value_at_risk_calibration.levels` to evaluate expected shortfall or extreme tail risk (e.g., 0.1% VaR).

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes.

**Strict Engineering Standards:** Adherence to PEP-8, strict static type hinting (`typing` module), technically detailed NumPy-standard docstrings, line-by-line in-text comments that explain the mathematical or logical purpose of every line (with the source equation from the LaTeX context where applicable), input validation, comprehensive error handling, and the fail-fast validation discipline of the manuscript are strictly required for all pull requests.

## Recommended Extensions

Future extensions, building upon this foundational framework, could include:

- **Alternative Backbone Architectures:** Replacing the shared MLP block with Temporal Convolutional Networks (TCNs) or Transformer encoders to capture long-range sequential dependencies in the feature space.
- **High-Frequency Data Integration:** Adapting the pipeline to ingest intraday order book imbalances, realized semivariances, or tick-level liquidity metrics as predictive features.
- **Alternative Regime Definitions:** Exploring macroeconomic indicators (e.g., credit spreads, yield curve slopes) or Natural Language Processing (NLP) sentiment scores as inputs to the gating network, moving beyond purely volatility-based regime states.
- **Multi-Task Learning:** Extending the residual experts to simultaneously predict volatility and expected shortfall, creating a unified risk-management objective.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original arXiv preprint:

```bibtex
@misc{ye2026regimegated,
  title={Regime-Gated Residual Mixture-of-Experts for Cross-Sectional Volatility Forecasting},
  author={Ye, Junyi and Borde, Gargi Vijay},
  howpublished={arXiv preprint arXiv:2608.12251 [q-fin.ST]},
  year={2026},
  url={https://arxiv.org/abs/2608.12251}
}
```

For the implementation itself, you may cite this repository:

```bibtex
@misc{chirinda2026rgresmoeimpl,
  author = {Chirinda, Craig S.},
  title = {Regime-Gated Residual Mixture-of-Experts for Cross-Sectional Volatility Forecasting: A Python Implementation},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/chirindaopensource/regime_gated_residual_mixture_of_experts_for_cross_sectional_volatility_forecasting}}
}
```

## Acknowledgments

- Credit to **Junyi Ye** and **Gargi Vijay Borde** for the foundational theoretical framework: the exact capacity-matched architectural ablations, the two-stage residual training protocol, the soft-gated regime routing mechanism, and the rigorous cross-market evaluation methodology.
- Data acknowledgment to **Yahoo Finance** for providing the historical U.S. and Japanese TSE Prime equity panels.
- This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, particularly the **PyTorch**, **NumPy**, **Pandas**, **SciPy**, and **Scikit-Learn** contributors.

--


*This README was generated based on the structure and content of the `regime_gated_residual_mixture_of_experts_for_cross_sectional_volatility_forecasting_draft.ipynb` notebook and follows best practices for research software documentation.*

