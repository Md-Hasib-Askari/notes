# Phase 5: Specialized Microbiology Tools - Exercise Sheet

## Exercise 1: Automated BLAST Search Pipeline
Write a script that takes a FASTA file with multiple sequences and performs automated BLAST searches against the NCBI nr database. Parse the results to extract the top 3 hits for each query sequence and save them to a CSV file with columns: query_id, hit_accession, e_value, bit_score, identity_percentage.

## Exercise 2: Primer Design Tool
Create a function that designs PCR primers for a given DNA sequence. The function should:
- Find regions with 40-60% GC content
- Ensure primer length between 18-25 nucleotides
- Check for primer dimers and hairpin structures
- Return forward and reverse primers with their melting temperatures

## Exercise 3: Restriction Enzyme Analysis
Develop a program that analyzes restriction enzyme cut sites in a DNA sequence. Given a sequence and a list of restriction enzymes, create a visualization showing:
- Cut site positions
- Fragment sizes
- A restriction map diagram
- Export results as both a plot and a data table

## Exercise 4: ORF Finder and Gene Prediction
Build an ORF (Open Reading Frame) finder that:
- Identifies all possible ORFs in all 6 reading frames
- Filters ORFs by minimum length (>300 bp)
- Predicts the most likely genes using codon usage bias
- Outputs gene coordinates, sequences, and predicted proteins in GFF format

## Exercise 5: QIIME2 Output Parser
Create a parser for QIIME2 diversity analysis output files. Write functions to:
- Read alpha diversity metrics from multiple samples
- Parse beta diversity distance matrices
- Generate summary statistics and visualizations
- Create a comparative report across different sample groups

## Exercise 6: Rarefaction Curve Generator
Implement a rarefaction analysis tool that:
- Takes an OTU table as input
- Generates rarefaction curves for each sample
- Calculates confidence intervals
- Determines optimal sequencing depth
- Creates publication-ready plots with error bars

## Exercise 7: Growth Curve Analysis and Modeling
Develop a comprehensive growth curve analyzer that:
- Reads plate reader data (Excel/CSV format)
- Fits logistic growth models to each well
- Calculates growth parameters (lag time, maximum growth rate, carrying capacity)
- Identifies outliers and performs quality control
- Generates individual and summary plots

## Exercise 8: Plate Reader Data Processor
Build a batch processor for 96-well plate reader data that:
- Handles multiple file formats (Tecan, BioTek, etc.)
- Performs background subtraction and normalization
- Calculates fold changes and statistical significance
- Creates heatmaps and dose-response curves
- Exports processed data with metadata

## Exercise 9: Automated Cell Counter
Create an image analysis pipeline for microscopy images that:
- Loads images from a directory
- Applies image preprocessing (noise reduction, contrast enhancement)
- Segments individual cells using watershed algorithm
- Counts cells and measures basic morphological parameters
- Generates a summary report with statistics and sample images

## Exercise 10: Colony Counting and Size Analysis
Develop a colony analysis tool for petri dish images that:
- Automatically detects and counts bacterial colonies
- Measures colony sizes and calculates size distributions
- Classifies colonies by morphology (round, irregular, etc.)
- Handles overlapping colonies
- Exports results as annotated images and data tables
