# SpatialTranscriptomicsAttention
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

Python dependencies managed through [`virtualenv`](https://virtualenv.pypa.io/en/latest/). After installing `virtualenv`,
create an environment with:

    virtualenv venv
    . venv/bin/activate

Then install dependencies with

    pip install -r requirements.txt

### Data Configuration

Raw data is not shipped with the software, and should be populated into
a local data folder prior to starting a run. We provide a handy script
to populate and/or check data contents against our stashed checksums.

    make configure

Will run these scripts.

