Accompanying repository for the paper [Probabilistic Calibration by Design for Neural Network Regression](https://arxiv.org/abs/2403.11964) (AISTATS 2024).

### Overview

This repository includes the implementation of Quantile Recalibration Training, a novel training procedure to learn predictive distributions that are probabilistically calibrated at every training step.
The full pipeline, including hyperparameter tuning and figures, is provided.
The implementation is based on the [repository](https://github.com/Vekteur/probabilistic-calibration-study) of the paper *A Large-Scale Study of Probabilistic Calibration in Neural Network Regression*.

The experiments are available in the directory [uq](/uq/). Additionally, a minimum PyTorch Lightning implementation of Quantile Recalibration Training, which can be easily adapted for custom datasets, is available in the directory [demo](/demo/).

<p align="center">
<img src="images/metric_per_dataset_per_epoch.svg?raw=true" alt="" width="88%" align="top">
<img src="images/diff_test_nll.svg?raw=true" alt="" width="88%" align="top">
</p>

### Abstract

Generating calibrated and sharp neural network predictive distributions for regression problems is essential for optimal decision-making in many real-world applications. To address the miscalibration issue of neural networks, various methods have been proposed to improve calibration, including post-hoc methods that adjust predictions after training and regularization methods that act during training. While post-hoc methods have shown better improvement in calibration compared to regularization methods, the post-hoc step is completely independent of model training. We introduce a novel end-to-end model training procedure called Quantile Recalibration Training, integrating post-hoc calibration directly into the training process without additional parameters. We also present a unified algorithm that includes our method and other post-hoc and regularization methods, as particular cases. We demonstrate the performance of our method in a large-scale experiment involving 57 tabular regression datasets, showcasing improved predictive accuracy while maintaining calibration. We also conduct an ablation study to evaluate the significance of different components within our proposed method, as well as an in-depth analysis of the impact of the base model and different hyperparameters on predictive accuracy.

### Installation

The experiments have been run in python 3.9 with the package versions listed in `requirements.txt`.

They can be installed using:
```
pip install -r requirements.txt
```

### Running the experiments

The main experiments, where metrics have been computed uniquely on the test set, can be run using:
```
python run.py name="full" nb_workers=1 repeat_tuning=5 \
        log_base_dir="logs" progress_bar=False \
        save_train_metrics=False save_val_metrics=False remove_checkpoints=True \
        selected_dataset_groups=["uci","oml_297","oml_299","oml_269"] \
        tuning_type="QRT"
```

Other experiments, where metrics have been computed during training on the training and validation sets, can be run using:
```
python run.py name="per_epoch" nb_workers=1 repeat_tuning=5 \
        log_base_dir="logs" progress_bar=False \
        save_train_metrics=True save_val_metrics=True remove_checkpoints=False \
        selected_dataset_groups=["uci","oml_297","oml_299","oml_269"] \
        tuning_type="QRT_per_epoch"
```

Then, the corresponding figures can be created in the notebook `create_figures.ipynb`.

### License

The license of the repository is MIT, except for the subdirectory uq/utils/fast_soft_sort, which is under the Apache 2.0 license.

### Citation

You can cite our paper using:
```
@inproceedings{DheurAISTATS2024,
  title     = {Probabilistic Calibration by Design for Neural Network Regression},
  author    = {Dheur, Victor and Ben taieb, Souhaib},
  booktitle = {The 27th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year      = {2024},
}
```
