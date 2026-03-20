# Table of contents
- [Introduction](#introduction)
- [Installation and usage](#installation-and-usage)
  - [1. Linux](#1-linux)
  - [2. Windows and Mac](#2-windows-and-mac)
- [Modules](#modules)
  - [1. CCM](#1-CCM)
  - [2. LOSDT](#2-LOSDT)
  - [3. MCSAlign](#3-MCSAlign)
- [References](#references)

# Introduction

This repository contains the setup to perform lead optimization with bioisosteres, other modules for a local energy minimization and aligment are available as well.

# Installation and usage

## 1. Linux

There are multiple options for Linux, including an executable file, a docker container or conda environment.

For the executable, you can download and run it. For the docker container, please download the latest LOSDT image from dockerhub and run the container.

In case you want to use a conda environment, run:

```
conda create -y -n LOSDT python=3.12
conda activate LOSDT
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
conda env update --file environment.yml
streamlit run Home.py
```

## 2. Windows and Mac

Install [docker.desktop](https://www.docker.com/products/docker-desktop/), search for the LOSDT image, pull it and run it with: Optional settings> Host port> 8501

After, you can open the local URL link and use the application.

# Modules

## 1. CCM

An OpenMM based energy minimization of complexes with a restrained environment, allowing relatively quick optimization of the ligand-protein-solvent complex. An option is provided to protonate the ligand using Dimorphite-DL in the range of pH 7.0-7.4. The following forcefields are used:

- Protein: ff19SB
- DNA: OL21
- Lipids: lipids21
- Waters: OPC3
- Small molecules: Sage 2.3.0 with AshGC neural network charge model

## 2. LOSDT

An open-source tool for lead design and optimization, built on the principle of safety by design. This tool serves as both an ADMET optimizer and an idea generator for navigating beyond patent space. It integrates multiple features, including ADMET-AI and ShEPhERD-score, leveraging a comprehensive bioisosteric reaction library for functionality.

Options for input are 2D compound or 3D ligand in bioactive confirmation inside protein complex. In case of the latter, the bioisosteric derivatives are automatically aligned to the input confirmation using the MCSAlign module. An option for both protonation with Dimorphite-DL at the range pH 7.0-7.4 and energy minimized with the CCM module are given.

## 3. MCSAlign

An RDKit based alignment tool using an iterative approach of finding Maximum Common Substructures between the 3D template and SMILES-derivatives, copy-pasting the common substructure and iteratively repeating on the leftover template.

# References

**[ADMET-AI](https://github.com/swansonk14/admet_ai):**

Swanson, K.; Walther, P.; Leitz, J.; Mukherjee, S.; Wu, J. C.; Shivnaraine, R. V.; Zou, J. 
ADMET-AI: A Machine Learning ADMET Platform for Evaluation of Large-Scale Chemical Libraries. 
*Bioinformatics* 2024.
[https://doi.org/10.1093/bioinformatics/btae416](https://doi.org/10.1093/bioinformatics/btae416)

---

**[ShEPhERD](https://github.com/coleygroup/shepherd-score):**

Adams, K.; Abeywardane, K.; Fromer, J.; Coley, C. W. ShEPhERD: diffusing shape, electrostatics, 
and pharmacophores for bioisosteric drug design. *Arxiv.org*.
[https://arxiv.org/html/2411.04130v1](https://arxiv.org/html/2411.04130v1)

---

**[Dimorphite-DL](https://github.com/durrantlab/dimorphite_dl):**

Ropp PJ, Kaminsky JC, Yablonski S, Durrant JD (2019) Dimorphite-DL: An open-source program for enumerating the ionization states of drug-like small molecules. J Cheminform 11:14. doi: [10.1186/s13321-019-0336-9](https://link.springer.com/article/10.1186/s13321-019-0336-9)

---

**[OpenMM](https://openmm.org/):**

A high-performance toolkit for molecular simulation. Use it as an application, a library, or a flexible programming environment. We include extensive language bindings for Python, C, C++, and even Fortran. 

---

**[RDKit](https://github.com/rdkit/rdkit):**

The RDKit is a collection of cheminformatics and machine-learning software written in C++ and Python.

---

**[pdb-tools](https://github.com/haddocking/pdb-tools):**

A swiss army knife for manipulating and editing PDB files.

---

**[streamlit](https://github.com/streamlit/streamlit):**

Python-based webserver builder.

---

**[streamlit-ketcher](https://github.com/mik-laj/streamlit-ketcher):**

Streamlit components that adds the ability to draw chemical compounds.

---

**[streamlit-molstar](https://github.com/pragmatic-streamlit/streamlit-molstar):**

Mol* is a modern web-based open-source toolkit for visualisation and analysis of large-scale molecular data.
