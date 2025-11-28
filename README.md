# DATE-GCN

The official implementation of “Dynamic Untangled Adjacency Learning with Adaptive Temporal Encoding for Skeleton-Based Action Recognition”.

# Dynamic Untangled Adjacency Learning

![alt text](https://github.com/SafwenNaimi/ATEM-GCN/blob/main/dynamic_untangled_scheme.png)



<p align="center"><strong>Figure 1:</strong> Proposed scheme for Dynamic Untangled Multi-Scale Adjacency learning (DUMA) module. Node colors reflect joint-level attention, and darker cells indicate stronger message passing. As the action "Right Hand Waving" unfolds, attention shifts toward the moving hand, demonstrating the model ability to adaptively capture motion-relevant joints across time and scale. $\hat{\mathbf{A}}_t^{(s)}$ is the attention-modulated adjacency matrix, with t being the frame number and s being the scale number.</p>


# Illustration of DATE-GCN

<p align="center">
  <img src="https://github.com/SafwenNaimi/ATEM-GCN/blob/main/ATEM-GCN_Design.png" alt="ATEM-GCNN visualization">
</p>

<p align="center"><strong>Figure 2:</strong> Our proposed approach: (a) is the overall model architecture, (b) is the DATE-GCN block, (c) is the dynamic untangled multi-scale adjacency (DUMA) module in the DATE-GCN, and (d) is the adaptive temporal encoding (ATEM) module in the DATE-GCN.</p>


# Preparation


    cd atemgcn
    conda env create -f atemgcn.yaml
    conda activate atemgcn
    pip install -e .


# Download Datasets

<em>1) There are 3 datasets to download:</em>

* NTU RGB+D 60 Skeleton
* NTU RGB+D 120 Skeleton
* NW-UCLA

<strong>NTU RGB+D 60 and 120</strong>

Request dataset here: [http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp](https://rose1.ntu.edu.sg/dataset/actionRecognition/)

Or download it from here: 

    [NTURGB+D 60]: https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_hrnet.pkl

    [NTURGB+D 120]: https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_hrnet.pkl

<strong>NW-UCLA</strong>

Download dataset from here: https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip?dl=0



<em>2) Move the downloaded datasets to:</em>

    - data/
      - NW-UCLA/
        - nw-ucla.pkl
      - ntu/
        - NTU-60/
          - ntu60.pkl
        - NTU-120/
          - ntu120.pkl


# Training & Testing

<strong>Training</strong>

* To train DATE-GCN on NTU RGB+D 60/120 with bone or motion modalities, modify line 4 in run.py and then execute it:


      # Example: training DATE-GCN on NTU RGB+D 120 cross subject under joint modality

      script_to_run = ['python', 'tools/train_model.py', 'configs/ntu120_xsub/j.py',
                 '--validate', '--test-last', '--test-best']
      
      python run.py


* To train DATE-GCN on NW-UCLA with bone or motion modalities, modify line 4 in run.py and then execute it:


      # Example: training DATE-GCN on NW-UCLA under bone motion modality

      script_to_run = ['python', 'tools/train_model.py', 'configs/nwucla/jm.py',
                 '--validate', '--test-last', '--test-best']
      
      python run.py

<strong>Test</strong>

Please check the configuration in the configs directory before testing DATE-GCN.

     # Example: testing DATE-GCN on NTU-60 X-Sub under the joint modality

     python tools/test.py configs/ntu60_xsub/j.py --eval mean_class_accuracy --checkpoint work_dirs/ntu60_xsub/j/latest.pth

To ensemble the results of different modalities, run the following command:

      python fusion.py
