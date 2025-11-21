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
