# MedScope

A lightweight benchmark of open-source large language models for medical question answering.

## Overview

This repository contains the code, figures, and experimental artifacts for benchmarking lightweight open-source LLMs on medical multiple-choice question answering.

Evaluated model families include:
- LLaMA
- Qwen
- Gemma

The benchmark focuses on:
- Accuracy
- Macro-F1
- Inference efficiency
- Invalid response rate
- Subject-wise performance
- Inter-model agreement
- Visualization-based analysis

## Repository Structure

- `scripts/`: benchmarking and plotting scripts
- `figures/`: main experimental figures
- `results/`: summarized benchmark results
- `docs/`: paper source files

## Main Figures

Representative figures include:
- multi-panel benchmark dashboard
- subject-wise clustered heatmap
- Pareto trade-off plot
- agreement heatmap
- Sankey visualization of prediction flow

## Notes

This repository focuses on lightweight, reproducible, and locally deployable open-source models for medical QA benchmarking.
