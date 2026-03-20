# Table of contents
- [Introduction](#introduction)
- [Installation and usage](#installation-and-usage)
  - [1. Linux](#1-linux)
  - [2. Windows](#2-windows)
- [Usage](#usage)
- [Modules](#modules)
  - [1. CCM](#1-CCM)
  - [2. LOSDT](#2-LOSDT)
  - [3. MCSAlign](#3-MCSAlign)

# Introduction

This repository contains the setup to perform lead optimization with bioisosteres, other modules for a local energy minimization and aligment are available as well.

# Installation and usage

## 1. Linux

There are multiple options for Linux, including an executable file, a docker container or conda environment.

For the executable, you can download and run it. For the docker container, please download the latest LOSDT image from dockerhub and run the container.

In case you want to use a conda environment, run:

```
conda env create -f environment.yml
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
conda activate LOSDT
streamlit run Home.py
```

## 2. Windows

Install [docker.desktop](https://www.docker.com/products/docker-desktop/), search for the LOSDT image, pull it and run it with: Optional settings> Host port> 8501

After, you can open the local URL link and use the application.

# Modules

## 1. CCM

An OpenMM based energy minimization of complexes with a restrained environment, allowing relatively quick optimization of the ligand-protein-solvent complex. An option is provided to protonate the ligand using Dimorphite-DL in the range of pH 7.0-7.4. The following forcefields are used:

- Protein: ff19SB
- DNA: OL21
- Lipids: lipids21
- Waters: OPC3
- Small molecules: openff Sage 2.3.0 with AshGC neural network charge model

## 2. LOSDT

An open-source tool for lead design and optimization, built on the principle of safety by design. This tool serves as both an ADMET optimizer and an idea generator for navigating beyond patent space. It integrates multiple features, including ADMET-AI and ShEPhERD-score, leveraging a comprehensive bioisosteric reaction library for functionality.

Options for input are 2D compound or 3D ligand in bioactive confirmation inside protein complex. In case of the latter, the bioisosteric derivatives are automatically aligned to the input confirmation using the MCSAlign module. An option for both protonation with Dimorphite-DL at the range pH 7.0-7.4 and energy minimized with the CCM module are given.

## 3. MCSAlign

An RDKit based alignment tool using an iterative approach of finding Maximum Common Substructures between the 3D template and SMILES-derivatives, copy-pasting the common substructure and iteratively repeating on the leftover template.