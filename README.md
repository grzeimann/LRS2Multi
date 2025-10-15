# LRS2Multi

**LRS2Multi** is a post-processing and extraction toolkit designed to operate on the reduced, flux-calibrated outputs from [**Panacea**](https://github.com/grzeimann/Panacea).  It provides higher-level combination, sky subtraction refinement, source extraction, and coaddition for the **LRS2 integral-field spectrograph** on the **Hobby–Eberly Telescope (HET)**.

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation and Setup](#installation-and-setup)
- [Reduction Workflow](#reduction-workflow)
- [Algorithm Summary](#algorithm-summary)
- [Data Products](#data-products)
- [Developer Pointers](#developer-pointers)
- [Citation](#citation)

---

## Overview

LRS2Multi extends *Panacea*’s reductions by combining multiple exposures, modeling residual backgrounds, and extracting science-ready 1D spectra.  
Where *Panacea* works from CCD frames to flux-calibrated fibers, **LRS2Multi** works from *fibers to final spectra* — aligning, scaling, and coadding exposures across the four LRS2 channels (UV, Orange, Red, Far-Red).

---

## Key Features

- **Seamless integration with Panacea outputs**  
  Ingests calibrated 2D fiber spectra and metadata directly from Panacea.

- **Residual background and sky subtraction**  
  Refines Panacea’s sky model with local annular background fitting, per-fiber residual correction, PCA residual fitting, and/or polynomial modeling.

- **Aperture and PSF-weighted source extraction**  
  Performs circular or custom extraction apertures, with adaptive weighting for partially resolved sources.

- **Exposure alignment and coaddition**  
  Aligns wavelength grids, scales exposures using mean/median values of common wavelength overlap regions, and coadds using inverse-variance weighting.

- **Quality-assurance products**  
  Generates diagnostic plots and intermediate FITS extensions to validate flux scaling and sky subtraction.

---

## Installation and Setup

### For TACC users

```bash
ssh username@ls6.tacc.utexas.edu
cd $HOME
ln -s $WORK work-ls6
module load python3
pip3 install astropy seaborn specutils scikit-learn --user
cd /work/XXXX/USERNAME/ls6
git clone https://github.com/grzeimann/LRS2Multi.git
```

Then, go to the visualization portal in a browser: https://vis.tacc.utexas.edu/jobs/

You can sign in and proceed to the utilities page by clinking the tab at the bottom or visiting: https://vis.tacc.utexas.edu/jobs/utils/

<p align="center">
  <img src="TACC_VIZ_utilities.png" width="650"/>
</p>

On the Utilities page, it the three bottom buttons to link Python3 and your directories for your work.

Then go back to the jobs page which should look like this:

<p align="center">
  <img src="TACC_VIZ_portal.png" width="650"/>
</p>

Request a job as shown in the attached image (just click submit when you pull up the same left hand settings). After a small wait time, a new screen will show up and you will click connect.  Sometimes there are not enough nodes initially and you have to wait a bit longer. After you connect, you should be in your work directory, which will allow you to navigate to LRS2Multi/notebooks.  There will be a file called example_reduction.ipynb.  Open that notebook and follow the instructions to get started.
