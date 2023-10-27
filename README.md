
# Brain Tumor Image Classification with Vision Transformers (ViT)

In this project, we delve into the compelling world of Vision Transformers (ViT) to tackle the challenging task of brain tumor image classification. While traditional Convolutional Neural Networks (CNNs) have been the go-to choice for image-related tasks, ViT introduces a novel perspective by harnessing the Transformer architecture, originally designed for natural language processing. We aim to create a custom ViT model capable of accurately categorizing brain MRI scans based on their content.

## About Vision Transformers (ViT)

Vision Transformers (ViT) are a pioneering deep learning architecture that applies the transformative Transformer model, originally crafted for sequential data like text, to image data. This innovative approach divides an input image into fixed-size patches, linearly embedding these patches into vectors. These embedded vectors are treated as sequences and processed by Transformer layers, enabling ViT to capture both local and global dependencies within the image.

## Project Motivation

Our project is motivated by the need for an effective solution to classify brain MRI scans into four crucial categories:

- **Glioma:** These tumors occur within the brain and/or spinal cord. Various types of gliomas include Astrocytomas, Ependymomas, and Oligodendrogliomas. Gliomas are among the most prevalent types of primary brain tumors.

- **Meningioma:** These tumors originate from the meninges, which are the membranes surrounding the brain and spinal cord. Meningiomas tend to grow slowly, often over an extended period without causing symptoms.

- **Pituitary Tumor:** These tumors form in the pituitary gland, a small gland located inside the skull. Most pituitary tumors are classified as pituitary adenomas, benign growths that typically do not spread beyond the skull.

- **No Tumor:** This category indicates MRI scans that do not show the presence of any brain tumor.

Accurate classification of brain MRI scans is critical for timely diagnosis and treatment. Our ViT-based model aims to address this challenge effectively.

## Dataset

The dataset for this project is sourced from Kaggle's [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). It comprises a collection of brain MRI scans from patients with and without brain tumors. Each image poses unique challenges due to varying sizes, resolutions, and contrasts. Our objective is to leverage the ViT architecture to develop a robust classification model capable of accurately identifying the presence of brain tumors in these MRI scans.

## Custom ViT Model Architecture

Within this project, we've meticulously designed a custom ViT model tailored for the task of brain tumor classification. The model features an initial patch embedding layer followed by multiple Transformer encoder layers. The self-attention mechanism of the Transformer empowers the model to learn intricate spatial relationships within the brain images, thus enhancing its classification accuracy.

## Project Structure

- `brain-tumor-using-vision-transformer.ipynb`: The central script containing code for data preprocessing, model creation, and training.
- `data/`: A directory housing the training and testing datasets.
- `models/`: A location to store model checkpoints.
- `results/`: A directory to store visualization results and metrics.

## Prerequisites

Before executing the code, ensure that you have the necessary libraries installed:

- TensorFlow
- TensorFlow Addons (TFA)
- NumPy
- Matplotlib
- Plotly
- Seaborn
- Scikit-Learn

You can install these libraries via `pip`:

```bash
pip install tensorflow tensorflow-addons numpy matplotlib plotly seaborn scikit-learn
```

## Usage

1. Clone this repository to your local machine.
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and organize it as follows:

```
- data/
  - Training/
    - glioma/
    - meningioma/
    - notumor/
    - pituitary/
  - Testing/
    - glioma/
    - meningioma/
    - notumor/
    - pituitary/
```

3. Run `brain-tumor-using-vision-transformer.ipynb` to preprocess the data, create and train the ViT model.

## Customization

Feel free to tailor this README to include additional details about your project, such as model evaluation, performance metrics, or any extra features you've implemented. You can also incorporate visualizations, examples, and explanations to make it more informative and engaging.

## Acknowledgments

If you found this project valuable or insightful, please consider giving it a star on GitHub. Your support and feedback are greatly appreciated!

## Author

- Noman Shabbir
- GitHub: [Noman Shabeer](https://github.com/NomanShabeer)
-

## Contact

If you have any questions, suggestions, or issues related to this project, please do not hesitate to contact me at [engr.nomanshabbir@gmail.com]

Thank you for your interest in our Brain Tumor Image Classification project with Vision Transformers!
