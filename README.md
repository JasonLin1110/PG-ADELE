# PG-ADELE (Prototype-Guided Adaptive Learning for Noisy Lable Correction)

## Installation 
◆ Install python dependencies.  
``` pip install -r requirement.txt ```

## SegThor
The Segthor dataset details and download can be found in [here](https://competitions.codalab.org/competitions/21145).   
For details on how noisy masks are generated, please refer to the [ADELE implementation](https://github.com/Kangningthu/ADELE/blob/master/SegThor/brat/loading.py).  
The training, validation, and testing datasets can be obtained separately by using the provided train.txt, valid.txt, and test.txt files.  
If the patient is in the training dataset, an additional noisy_masks will be created to store the generated noisy masks. During training, only the data with noisy masks will be used for training, and the original masks will not be used.  
We uniformly resize all images and masks to 256×256.  
The dataset is stored as follows:  
```plaintext
data_dir/
    └─ Patient_01/
        └─ images/
            ├─ slice_1.png
            ├─ slice_2.png
            └─ ...
        └─ masks/
            ├─ slice_1.png
            ├─ slice_2.png
            └─ ...
        └─ noisy_masks/
            ├─ slice_1.png
            ├─ slice_2.png
            └─ ...
    └─ Patient_02/
        └─ images/
            ├─ slice_1.png
            ├─ slice_2.png
            └─ ...
        └─ masks/
            ├─ slice_1.png
            ├─ slice_2.png
            └─ ...
        └─ noisy_masks/
            ├─ slice_1.png
            ├─ slice_2.png
            └─ ...

    ...

    └─ Patient_23/
        └─ images/
            ├─ slice_1.png
            ├─ slice_2.png
            └─ ...
        └─ masks/
            ├─ slice_1.png
            ├─ slice_2.png
            └─ ...

    ...
```
The arguments represent:   
```
parser.add_argument("--data_dir", type=str, default="SegThor/train")
parser.add_argument("--correct_dir", type=str, default="correct")
parser.add_argument("--save_dir", type=str, default="PGADELE")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--momentum", type=float, default=0.99)
parser.add_argument("--number_epochs", type=int, default=50)
parser.add_argument("--correct_freq", type=int, default=10)
parser.add_argument("--class_num", type=int, default=5)
parser.add_argument("--channel", type=int, default=1)

# ADELE + label correction 
parser.add_argument("--beta_fg", type=float, default=0.8, help="confidence threshold for the foreground")
parser.add_argument("--beta_bg", type=float, default=0.95, help="confidence threshold for the background")
parser.add_argument("--delta", type=float, default=0, help="similarity threshold(-1~1)")
parser.add_argument("--r", type=float, default=0.9, help="the r for the label correction")
parser.add_argument("--effect_learn", type=float, default=0.1, help="the effect leraning threshold")

# prototype set
parser.add_argument("--p_num", type=int, default=5, help="prototypes number of each class")

# loss parameters
parser.add_argument("--rho", type=float, default=0.8, help='the threshold when select the target for JSD')
parser.add_argument("--alpha", type=float, default=0.01, help='loss weight for prototype learning (HAPMC+PPA)')
parser.add_argument("--m", type=float, default=20, help='HAPMC positive margin (degree)')
parser.add_argument("--n_m", type=float, default=0.5, help='HAPMC negative margin (0~1)')
parser.add_argument("--tau", type=float, default=0.01, help='HAPMC sharp')
parser.add_argument("--eps", type=float, default=0.0, help='hard prototype selection threshold')
```
Baseline parameters follows default setting.  
Training experiement: ``` python train_segthor.py ```  
Testing: ``` python test_segthor.py ```


## ISIC 2017
The ISIC 2017 dataset details and download can be found in [here](https://challenge.isic-archive.com/data/)  
For details on how noisy masks are generated, thanks to the work of Kangning Liu, please refer to the [Learning to Segment from Noisy Annotations: A Spatial Correction Approach](https://github.com/michaelofsbu/SpatialCorrection/tree/main) I3 method.  
We uniformly resize all images and masks to 256×256.  
The dataset is stored as follows:  
```plaintext
data_dir/
    └─ Training_resize/
        ├─ ISIC_xxxxxxx.jpg
        ├─ ISIC_xxxxxxo.jpg
        |  ...
        └─ ISIC_ooooooo.jpg
    └─ Training_GT_resize/
        ├─ ISIC_xxxxxxx_segmentation.png
        ├─ ISIC_xxxxxxo_segmentation.png
        |  ...
        └─ ISIC_ooooooo_segmentation.png
    └─ Validation_resize/
        ├─ ISIC_xxxxxxx.jpg
        ├─ ISIC_xxxxxxo.jpg
        |  ...
        └─ ISIC_ooooooo.jpg
    └─ Validation_GT_resize/
        ├─ ISIC_xxxxxxx_segmentation.png
        ├─ ISIC_xxxxxxo_segmentation.png
        |  ...
        └─ ISIC_ooooooo_segmentation.png
    └─ Test_resize/
        ├─ ISIC_xxxxxxx.jpg
        ├─ ISIC_xxxxxxo.jpg
        |  ...
        └─ ISIC_ooooooo.jpg
    └─ Test_GT_resize/
        ├─ ISIC_xxxxxxx_segmentation.png
        ├─ ISIC_xxxxxxo_segmentation.png
        |  ...
        └─ ISIC_ooooooo_segmentation.png
```
The arguments represent:   
```
parser.add_argument("--data_dir", type=str, default="ISIC2017")
parser.add_argument("--correct_dir", type=str, default="correct_ISIC")
parser.add_argument("--save_dir", type=str, default="PGADELE_ISIC")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--number_epochs", type=int, default=50)
parser.add_argument("--correct_freq", type=int, default=10)
parser.add_argument("--class_num", type=int, default=2)
parser.add_argument("--channel", type=int, default=3)

# ADELE + label correction 
parser.add_argument("--beta_fg", type=float, default=0.95, help="confidence threshold for the foreground")
parser.add_argument("--beta_bg", type=float, default=0.95, help="confidence threshold for the background")
parser.add_argument("--delta", type=float, default=0, help="similarity threshold(-1~1)")
parser.add_argument("--r", type=float, default=0.9, help="the r for the label correction")
parser.add_argument("--effect_learn", type=float, default=0.01, help="the effect leraning threshold")

# prototype set
parser.add_argument("--p_num", type=int, default=5, help="prototypes number of each class")

# loss parameters
parser.add_argument("--rho", type=float, default=0.8, help='the threshold when select the target for JSD')
parser.add_argument("--alpha", type=float, default=0.01, help='loss weight for prototype learning (HAPMC+PPA)')
parser.add_argument("--m", type=float, default=20, help='HAPMC positive margin (degree)')
parser.add_argument("--n_m", type=float, default=0.5, help='HAPMC negative margin (0~1)')
parser.add_argument("--tau", type=float, default=0.01, help='HAPMC sharp')
parser.add_argument("--eps", type=float, default=0.0, help='hard prototype selection threshold')
```
Baseline parameters follows default setting.  
Training experiement: ``` python train_ISIC.py ```  
Testing: ``` python test_ISIC.py ```

## Thanks
Thanks to the work of Kangning Liu, the code of this repository draws heavily from his [ADELE repository](https://github.com/Kangningthu/ADELE/tree/master).  
Thanks to the work of Tianfei Zhou, the concept of his [ProtoSeg repository](https://github.com/tfzhou/ProtoSeg) served as the foundation for the prototype refinement component of this repository.