# TAZIA
Tabelamento
Automático de baixo
Z usando
Inteligencia
Artificial

This project was made as a final project for the course on Artificial Intelligence of 
the DCA - UFRN and is not a astrophysical exploration of the dataset.

## Data sources
Download all resources to the `data/` folder.
- Images from: [Zenodo - Galaxy Zoo 2: Images from Original Sample](https://zenodo.org/records/3565489#.Y3vFKS-l0eY);
- Data from full catalog (used for the labels): [Galaxy Zoo 2 data](https://data.galaxyzoo.org/#section-7) 
([Table 1](https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz));
- [Column descriptions for Table 1](https://gz2hart.s3.amazonaws.com/gz2_hart16.txt).

### Data reduction
We dropped most of the columns and some of the objects, as described in `classes.md`. 
The script for this filtering is `scripts/clear_labels.py`.

A small visualization of the dataset was made with the `scripts/class_plots.py`. 

## Zoobot encoder
We implemented grouping and classification using the `hf_hub:mwalmsley/zoobot-encoder-convnext_nano`
encoder from zoobot. We opted to separate image encoding from feature grouping/classification,
saving the feature matrix in a `csv`. The pipeline for grouping and classification is as follows:

Firstly, obtain feature matrix from a subset of the full catalog, in `zoobot_encoder.py`:
1. Configure the directory where the images are stored in `src_root`;
2. Select the size of the subset to encode in `n`;
3. Define the size of the batch of partial saving;

For the analysis:
- Grouping using k-means and t-SNE, run `grouping_tsne.py`;
- Grouping using k-means and PCA, run `grouping_pca.py`;
- Classification using 2 single dense layer perceptrons, run `zoobot_classifier.py`

The results should be saved in `results/zoobot-encoder-convnext_nano/`.