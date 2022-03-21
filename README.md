# GATSBy
## *G*raph *AT*tention for *S*patial *B*iolog*y*
Tools to infer local feature dependencies in spatial transcriptomics datasets

All code developed for use on \*NIX systems with
CUDA capability.

## Introduction / Background

## Data

## Framework Overview

## Usage


### Environment Setup

The main toolset is built in python and machine learning components
are implemented on [`PyTorch`](https://pytorch.org/).

Python dependencies managed through [`conda`]. After installing `conda`, create an environment with:

    conda create -n gatsby python=3.8

Then activate the environment with

    conda activate gatsby

### Data Configuration

Raw data is not shipped with the software, and should be populated into
a local data folder prior to starting a run. We provide a handy script
to populate and/or check data contents against our stashed checksums.

    make configure

Will run these scripts.

