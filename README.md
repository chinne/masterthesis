Deep generative models as an approach to generate new artificial tabular data
==============================================================================

Using Generative Adversarial Networks, Wasserstein GAN and Variational Autoencoder as oversampling method to generate new artificial data.

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Savings of the trained models
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── gan.py
        │   ├── cgan.py
		│	├── wgan.py
        │	├── cwgan.py
		│	└── vae.py
		│
        └── common  <- Scripts to help out
            ├── accuracy_XGboost.py
			└── helperfunctions.py
