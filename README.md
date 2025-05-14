# ATEM-GCN

The official implementation of “ATEM-GCN: Unified Graph-Based Architecture With Dynamic Untangled Adjacency and Adaptive Temporal Encoding for Human Activity Recognition”.

# Dynamic Untangled Adjacency Learning

<p align="center">
  <img src="https://github.com/SafwenNaimi/ATEM-GCN/blob/main/dynamic_untangled_scheme.png" alt="ATEM-GCN visualization">
</p>

<p align="center"><strong>Figure 1:</strong> Proposed dynamic untangled multi-scale aggregation scheme. Darker cells indicate stronger message passing. Node colors reflect joint-level attention, and matrix intensities indicate reweighted spatial dependencies. As the action "Hand Waving" unfolds, attention shifts toward the moving hand, demonstrating the model's ability to adaptively capture motion-relevant joints across time and scale.</p>


# Illustration of ATEM-GCN

<p align="center">
  <img src="https://github.com/SafwenNaimi/ATEM-GCN/blob/main/ATEM-GCN_Design.png" alt="ATEM-GCNN visualization">
</p>

<p align="center"><strong>Figure 2:</strong> Our proposed approach: (a) is the overall model architecture, (b) is the ATEM-GCN block, (c) is the dynamic untangled adjacency module in the ATEM-GCN, and (d) is the adaptive temporal encoding module in the ATEM-GCN.</p>


# Preparation


    cd atemgcn
    conda env create -f atemgcn.yaml
    conda activate atemgcn
    pip install -e .


# Download datasets

There are 3 datasets to download:
* NTU RGB+D 60 Skeleton
* NTU RGB+D 120 Skeleton
* NW-UCLA
