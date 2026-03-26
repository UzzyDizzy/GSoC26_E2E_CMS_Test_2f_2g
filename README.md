# ML4SCI GSoC 2026 
## Specific Tasks 2f and 2g: Deep Learning Inference for Mass Regression

## 📌 Project Overview

This repository acts as the central point for the technical evaluation of tasks pertaining to the machine learning extraction of fundamental mass properties from high-granularity spatial collision signals inside the CMS detector.

The implemented solution is fully embodied in the `ML4SCI_Test_2f_2g.ipynb` Jupyter Notebook, designed completely modularly bridging localized PyTorch architectures and CMS Software Framework (CMSSW) deployments.

---

## 🚀 Task 2f: Deep Learning Inference for Mass Regression

**Objective:** Train an end-to-end Convolutional Neural Network (CNN) to accurately regress the invariant mass (`am`) of a particle utilizing structured spatial image matrices.

### 📊 Dataset Details
The fundamental representations supplied within the Parquet files format isolated `125x125` visual matrices across an `ieta` and `iphi` grid structure over four sub-detector signatures `X_jet` (`Track pT`, `DZ`, `D0`, `ECAL`).

- **Original Datasets Source**: [CERNBox Data Source](https://cernbox.cern.ch/s/zUvpkKhXIp0MJ0g)
- **Target Constraint**: The feature target variable dictates estimating the final particle mass profile `am`.
- **Feature Filtering**: Per exact task methodology guidelines, the input pipeline explicitly constrains the network to exploit solely the `ECAL` and `Track pT` kinematics channels. Other variables are inherently shielded across the normalization layers.

### 🧠 Deep Learning Architecture & Training
Informed by continuous distribution diagnostics and outlier isolation protocols:
- **Baseline Models**: Created both a modular `SmallCNN` and a fully regularized `CustomCNN` combining robust staggered $3\times3$ Convolutional paths, Dropout (rate=0.3), and strategic Batch Normalization methodologies.
- **Physics Normalization**: Employed advanced Heavy-Tailed target scaling (`np.log1p`) alongside isolated static noise thresholding.
- **Cross-Validation Control**: Guaranteed a strict and exact split of **80% Training Data** and **20% Validation Data**.
- **Combatting Overfitting**: By relying thoroughly on the **SmoothL1Loss** criterion (for gradient-proof inflection point variance algorithms) combined with strategic `ReduceLROnPlateau` scheduling and dropout implementations, the network completely mitigates validation data memorization strategies (over-fitting).

*(For exact loss evaluation and predictions scaling visuals, refer strictly to the diagnostic matplotlib plots at the end of the `ML4SCI_Test_2f_2g.ipynb` file).*

---

## ⚙️ Task 2g: Inference within CMSSW

**Objective:** Validate model performance inside CERN's localized Linux container environments via automated ONNX operations running entirely within CMSSW over ROOT outputs.

### 1. Model Serialization (Task 1 to ONNX)
To enforce standardized cross-framework interoperability, the successfully converged deep regression PyTorch prototype is fundamentally exported out of the script runtime directly leveraging ONNX protocols:
```python
# The model creates an automated ONNX graph map off an operational dummy tensor
import torch.onnx
dummy_input = torch.randn(1, 4, 125, 125).to(device)
torch.onnx.export(model, dummy_input, "sample.onnx", opset_version=11)
```

### 2. CMSSW Container Setup
Following execution methodologies strictly, we establish an isolated CentOS7 standalone development target leveraging the specific `cc7-cmssw-cvmfs` Docker images built from CERN’s live repository systems.

**Bootstrapping Environment:**
```bash
# Enter CMSSW Patch 11 development areas
cmsrel CMSSW_12_0_2
cd CMSSW_12_0_2/src/
cmsenv
```

### 3. Sub-Module Integration & Compilation
Prior to mapping target limits, explicit Reco pipelines from `DataFormats` are necessary. 
```bash
# Add specialized packages
git cms-addpkg DataFormats/TestObjects

# Clone End-to-End specific Reco hooks
git clone https://github.com/rchudasa/RecoE2E.git

# Initialize global compilation threading
scram b -j8 # used -j1 as my RAM could not support j8
```

### 4. Running E2E Execution & Verification
To test inferences across simulated validation scenarios, the `E/gamma` root binaries serve as the explicit testing source:
- **E/gamma Source**: [Download ROOT Target](https://cernbox.cern.ch/s/Yp3oZl8cUU6JoFC) 

Finally, with all targets aligned locally (both the ROOT inference package `SIM_DoubleGammaPt50_Pythia8_1000Ev.root` and the `sample.onnx` neural mapping), execute:

```bash
# Run End-To-End Pythia8 Double Gamma inference timing simulation
cmsRun RecoE2E/EGTagger/python/EGInference_cfg.py inputFiles=file:SIM_DoubleGammaPt50_Pythia8_1000Ev.root maxEvents=-1 EGModelName=sample.onnx
```

### 5. Results and Logs
Finally, after running inference command, we can run `cat resources.json` to get the results which I have also saved in `resources.json` and below are the inference logs:

```bash
# Logs for the inference timing simulation
=============================================

MessageLogger Summary

 type     category        sev    module        subroutine        count    total
 ---- -------------------- -- ---------------- ----------------  -----    -----
    1 fileAction           -s file_close                             1        1
    2 fileAction           -s file_open                              2        2

 type    category    Examples: run/evt        run/evt          run/evt
 ---- -------------------- ---------------- ---------------- ----------------
    1 fileAction           PostGlobalEndRun
    2 fileAction           pre-events       pre-events

Severity    # Occurrences   Total Occurrences
--------    -------------   -----------------
System                  3                   3

dropped waiting message count 0

real    2m50.740s
user    3m6.632s
sys     0m15.757s
```

---

**Mentors:**
Sergei Gleyzer (University of Alabama) | Ruchi Chudasama (University of Alabama) | Shravan Chaudhari (New York University) | Purva Chaudhari (Vishwakarma Institute of Technology)

*Submitted as part of the GSoC 2026 ML4SCI candidate evaluation.*
