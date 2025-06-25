<!-- # cv_miniProject2submit

Width <= 512px: 3630/3680 (98.64%)
Height <= 512px: 3648/3680 (99.13%) -->

# Oxford Pet-III Segmentation

This repository contains training and inference pipelines for segmentation tasks on the Oxford Pet-III dataset. Supported models include U-Net, Autoencoder, CLIP, and Prompt-based segmentation at training-ready states.

## 📄 Report  

If you're interested in the **project highlights**, **training results**, and some **visualisations**, feel free to check out the full report:  
👉 [Image Segmentation Report (PDF)](https://github.com/X-jy-liu/CatDogSegNet/blob/main/Image%20Segmentation%20Report.pdf)

## 📦 Setup

### 0. Clone the Repository and Environment Setup

```bash
git clone git@github.com:X-jy-liu/CatDogSegNet.git
cd <your-repo-name>
```

Set up the environment (using conda):
```bash
conda create -n petsegmentation python==3.10
conda activate petsegmentation
pip install -r requirements.txt
```

### 1. Download the Dataset

Download the [**Oxford-IIIT Pet Dataset**](https://www.robots.ox.ac.uk/~vgg/data/pets/) and place it in the project root directory with the folder name exactly as `Dataset`.  
The directory structure should look like:

```
Dataset/
├── Test/
├── TrainVal/
└── ...
```

---

## 🚀 Training

All training is initiated via `src/main.py`.

### Available modes:

| Mode | Model         |
|------|---------------|
| 0    | U-Net         |
| 1    | Autoencoder   |
| 2    | CLIP          |
| 3    | Prompt-based  |

### Example:

```bash
python src/main.py --mode 0 --epochs 200 --patience 5 | tee unet_segmentation.log
```

More examples can be found in [`src/run_training.sh`](src/run_training.sh)

---

### Prompt-based Training

1)  Preprocess and Generate the Augmented Prompt Dataset

```bash
python src/preprocess_data_with_prompt.py --mode 0 \
    --img_dir <train_image_folder> \
    --msk_dir <train_mask_folder> \
    --output_dir <processed_output_folder>
```
This generates the image-mask-prompt pairs in this structure. There should be ~20,000 image-mask-prompt pairs.
```
Dataset/
├── ...
├── ProcessedWithPrompt/ 
└── ...               ├── color/
                            points/
                      └── label/
```

2) Run the prompt-based training. This will save the temporary best checkpoint in `./param/tmp_prompt_checkpoint/`. At the end, global best performing model config and training plot will be saved.
``` bash
# Train prompt segmentation model
python src/main.py  --img_dir Dataset/ProcessedWithPrompt/color/ \
                    --msk_dir Dataset/ProcessedWithPrompt/label/ \
                    --pnt_dir Dataset/ProcessedWithPrompt/color/points/ \
                    --mode 3 \
                    --epochs 100
                    --patience 10
```

---

## 🔍 Inference

After training, you can run inference using:

```bash
python src/run_inference.py \
    --image_dir <test_image_folder> \
    --mask_dir <test_label_folder> \
    --save_dir <prediction_output_folder> \
    --checkpoint_path <path_to_model_checkpoint> \
    --device cuda \
    --mode 0
```

This saves the predicted masks for later IoU evaluation.

To check all available options:

```bash
python src/run_inference.py -h
```
### Prompt-based Inference
1）Sample prompt points for the test dataset
```bash
# generate test prompt points ONLY
python src/preprocess_data_with_prompt.py --mode 1 \
    --img_dir <test_image_folder> \
    --msk_dir <test_label_folder> \
    --output_dir <processed_test_output_folder>

```
2) Read the test images, label and sampled prompt point to run inference 
```bash
python src/run_inference.py \
    --image_dir <processed_test_output_folder>/color \
    --mask_dir <processed_test_output_folder>/label \
    --point_dir <processed_test_output_folder>/points \
    --save_dir <prompt_prediction_output_folder> \
    --checkpoint_path <path_to_prompt_model_checkpoint> \
    --target_size 512 \
    --device cuda \
    --mode 3
```

---

## 📊 IoU Calculation

To evaluate the performance using Intersection-over-Union (IoU):
```bash
python src/calculate_IoU.py \
    --gt_folder <ground_truth_mask_folder> \
    --pred_folder <predicted_mask_folder> \
    --output_file <output_iou_results_file>
```

Example:
```bash
python src/calculate_IoU.py \
    --gt_folder Dataset/Test/label \
    --pred_folder Dataset/Test/unet_preds \
    --output_file unet_iou.txt
```

This will save the IoU results to `unet_iou.txt` for later analysis.

## 📈 Results

The best-performing model (U-Net) achieves an IoU of **0.7992** on the cat–dog–background segmentation task.
We further evaluate robustness under various perturbations and extend the architecture to support prompt-based interactive segmentation using points or boxes.

## 📜 License

This project is licensed under the Apache 2.0 License. See the `LICENSE` file for details.

## 🌟 Acknowledgments
- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [U-Net: Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- [CLIP: Radford et al., 2021](https://arxiv.org/abs/2103.00020)
- Special thanks to [@Jinrusui](https://github.com/Jinrusui) for his contributions to this project.
