# Package Development

## Creating Installable Python Packages
Develop reusable Python packages for microbiology research tools.

```python
# setup.py - Package configuration file
from setuptools import setup, find_packages

setup(
    name="microbio-toolkit",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@university.edu",
    description="A toolkit for microbiology data analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/microbio-toolkit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "biopython>=1.79",
        "matplotlib>=3.3.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "microbio-analyze=microbio_toolkit.cli:main",
        ],
    },
)
```

```python
# microbio_toolkit/__init__.py - Package initialization
"""
Microbio Toolkit - A Python package for microbiology data analysis

This package provides tools for:
- Sequence analysis
- Growth curve fitting
- Microbiome data processing
- Laboratory data management
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@university.edu"

# Import main modules
from .sequence_analysis import SequenceAnalyzer, calculate_gc_content
from .growth_curves import GrowthCurveAnalyzer, fit_logistic_model
from .microbiome import MicrobiomeAnalyzer, calculate_diversity
from .utils import load_fasta, save_results

__all__ = [
    "SequenceAnalyzer",
    "calculate_gc_content", 
    "GrowthCurveAnalyzer",
    "fit_logistic_model",
    "MicrobiomeAnalyzer",
    "calculate_diversity",
    "load_fasta",
    "save_results"
]
```

```python
# microbio_toolkit/sequence_analysis.py - Main sequence analysis module
"""Sequence analysis tools for microbiology research."""

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from typing import List, Dict, Union

class SequenceAnalyzer:
    """Comprehensive sequence analysis for microbial genomes."""
    
    def __init__(self, sequences: Union[str, List[str]] = None):
        """
        Initialize sequence analyzer.
        
        Args:
            sequences: FASTA file path or list of sequence strings
        """
        self.sequences = []
        
        if sequences:
            self.load_sequences(sequences)
    
    def load_sequences(self, sequences: Union[str, List[str]]):
        """Load sequences from file or list."""
        
        if isinstance(sequences, str):
            # Load from FASTA file
            self.sequences = [str(record.seq) for record in SeqIO.parse(sequences, "fasta")]
        elif isinstance(sequences, list):
            self.sequences = sequences
        else:
            raise ValueError("Sequences must be file path or list of strings")
    
    def calculate_gc_content(self, sequence: str = None) -> float:
        """Calculate GC content for a sequence."""
        
        if sequence is None:
            if not self.sequences:
                raise ValueError("No sequences loaded")
            sequence = self.sequences[0]
        
        sequence = sequence.upper()
        gc_count = sequence.count('G') + sequence.count('C')
        total_count = len(sequence)
        
        return (gc_count / total_count * 100) if total_count > 0 else 0.0
    
    def find_orfs(self, sequence: str = None, min_length: int = 100) -> List[Dict]:
        """Find open reading frames in sequence."""
        
        if sequence is None:
            if not self.sequences:
                raise ValueError("No sequences loaded")
            sequence = self.sequences[0]
        
        seq_obj = Seq(sequence)
        orfs = []
        
        # Check all 6 reading frames
        for strand, seq in [(1, seq_obj), (-1, seq_obj.reverse_complement())]:
            for frame in range(3):
                translated = seq[frame:].translate()
                
                # Find ORFs between start and stop codons
                start = 0
                while start < len(translated):
                    start_pos = translated.find('M', start)
                    if start_pos == -1:
                        break
                    
                    stop_pos = start_pos
                    while stop_pos < len(translated) and translated[stop_pos] != '*':
                        stop_pos += 1
                    
                    orf_length = (stop_pos - start_pos) * 3
                    if orf_length >= min_length:
                        orfs.append({
                            'strand': strand,
                            'frame': frame + 1,
                            'start': start_pos * 3 + frame,
                            'end': stop_pos * 3 + frame,
                            'length': orf_length,
                            'protein': str(translated[start_pos:stop_pos])
                        })
                    
                    start = start_pos + 1
        
        return sorted(orfs, key=lambda x: x['length'], reverse=True)
    
    def analyze_composition(self, sequence: str = None) -> Dict:
        """Analyze nucleotide composition."""
        
        if sequence is None:
            if not self.sequences:
                raise ValueError("No sequences loaded")
            sequence = self.sequences[0]
        
        sequence = sequence.upper()
        total_length = len(sequence)
        
        composition = {
            'A': sequence.count('A') / total_length * 100,
            'T': sequence.count('T') / total_length * 100,
            'G': sequence.count('G') / total_length * 100,
            'C': sequence.count('C') / total_length * 100,
            'N': sequence.count('N') / total_length * 100
        }
        
        composition['GC_content'] = composition['G'] + composition['C']
        composition['AT_content'] = composition['A'] + composition['T']
        
        return composition

def calculate_gc_content(sequence: str) -> float:
    """Standalone function to calculate GC content."""
    analyzer = SequenceAnalyzer()
    return analyzer.calculate_gc_content(sequence)
```

## Documentation with Sphinx
Create comprehensive documentation for your package.

```python
# docs/conf.py - Sphinx configuration
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'Microbio Toolkit'
copyright = '2024, Your Name'
author = 'Your Name'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'biopython': ('https://biopython.org/docs/latest/api/', None),
}
```

```rst
# docs/index.rst - Main documentation file
Microbio Toolkit Documentation
==============================

A comprehensive Python toolkit for microbiology data analysis.

Features
--------

* Sequence analysis and ORF finding
* Growth curve fitting and analysis
* Microbiome diversity calculations
* Laboratory data management tools

Installation
------------

Install using pip:

.. code-block:: bash

    pip install microbio-toolkit

Or install from source:

.. code-block:: bash

    git clone https://github.com/yourusername/microbio-toolkit.git
    cd microbio-toolkit
    pip install -e .

Quick Start
-----------

.. code-block:: python

    from microbio_toolkit import SequenceAnalyzer
    
    # Analyze DNA sequence
    analyzer = SequenceAnalyzer()
    gc_content = analyzer.calculate_gc_content("ATGCGATCGTAGC")
    print(f"GC Content: {gc_content:.2f}%")

API Reference
=============

.. toctree::
   :maxdepth: 2
   
   api/sequence_analysis
   api/growth_curves
   api/microbiome
   api/utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```

## Testing with pytest
Implement comprehensive testing for your package.

```python
# tests/test_sequence_analysis.py - Unit tests
import pytest
import numpy as np
from microbio_toolkit.sequence_analysis import SequenceAnalyzer, calculate_gc_content

class TestSequenceAnalyzer:
    """Test cases for SequenceAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SequenceAnalyzer()
        self.test_sequence = "ATGCGATCGTAGCAAA"
    
    def test_calculate_gc_content(self):
        """Test GC content calculation."""
        expected_gc = 50.0  # 8 G/C out of 16 nucleotides
        result = self.analyzer.calculate_gc_content(self.test_sequence)
        assert abs(result - expected_gc) < 0.01
    
    def test_calculate_gc_content_empty_sequence(self):
        """Test GC content with empty sequence."""
        result = self.analyzer.calculate_gc_content("")
        assert result == 0.0
    
    def test_analyze_composition(self):
        """Test nucleotide composition analysis."""
        composition = self.analyzer.analyze_composition(self.test_sequence)
        
        # Check that percentages sum to ~100%
        total = composition['A'] + composition['T'] + composition['G'] + composition['C']
        assert abs(total - 100.0) < 0.01
        
        # Check GC content matches
        assert abs(composition['GC_content'] - 50.0) < 0.01
    
    def test_find_orfs(self):
        """Test ORF finding functionality."""
        # Test sequence with known ORF
        test_seq = "ATGAAATTTAAATAG"  # Start codon + sequence + stop codon
        orfs = self.analyzer.find_orfs(test_seq, min_length=9)
        
        assert len(orfs) >= 1
        assert orfs[0]['length'] >= 9
        assert 'protein' in orfs[0]
    
    def test_load_sequences_list(self):
        """Test loading sequences from list."""
        sequences = ["ATGC", "GCTA", "TTAA"]
        self.analyzer.load_sequences(sequences)
        
        assert len(self.analyzer.sequences) == 3
        assert self.analyzer.sequences[0] == "ATGC"
    
    def test_no_sequences_loaded_error(self):
        """Test error when no sequences are loaded."""
        empty_analyzer = SequenceAnalyzer()
        
        with pytest.raises(ValueError, match="No sequences loaded"):
            empty_analyzer.calculate_gc_content()

def test_standalone_gc_function():
    """Test standalone GC content function."""
    result = calculate_gc_content("ATGC")
    expected = 50.0  # 2 G/C out of 4 nucleotides
    assert abs(result - expected) < 0.01

class TestSequenceAnalyzerIntegration:
    """Integration tests for SequenceAnalyzer."""
    
    @pytest.fixture
    def sample_sequences(self):
        """Provide sample sequences for testing."""
        return [
            "ATGCGATCGTAGCAAATAG",  # Contains ORF
            "GCTAGCTAGCTAGC",        # High GC content
            "AAAAAATTTTTTTT"         # Low GC content
        ]
    
    def test_batch_analysis(self, sample_sequences):
        """Test analysis of multiple sequences."""
        analyzer = SequenceAnalyzer(sample_sequences)
        
        results = []
        for seq in analyzer.sequences:
            gc_content = analyzer.calculate_gc_content(seq)
            composition = analyzer.analyze_composition(seq)
            orfs = analyzer.find_orfs(seq)
            
            results.append({
                'gc_content': gc_content,
                'composition': composition,
                'num_orfs': len(orfs)
            })
        
        assert len(results) == 3
        assert all('gc_content' in result for result in results)

# tests/conftest.py - Shared test configuration
import pytest
import tempfile
import os

@pytest.fixture
def temp_fasta_file():
    """Create temporary FASTA file for testing."""
    fasta_content = """>seq1
ATGCGATCGTAGCAAATAG
>seq2  
GCTAGCTAGCTAGC
>seq3
AAAAAATTTTTTTT
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(fasta_content)
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    if os.path.exists(temp_file):
        os.unlink(temp_file)

# Run tests with coverage
# pytest tests/ --cov=microbio_toolkit --cov-report=html
```

## Version Control with Git
Best practices for version control in package development.

```bash
# .gitignore - Files to ignore in version control
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
*.manifest
*.spec

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Documentation
docs/_build/

# Environment
.env
.venv
env/
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

```python
# setup_dev_environment.py - Development environment setup
"""Set up development environment for microbio-toolkit."""

import subprocess
import sys
import os

def run_command(command, description):
    """Run shell command with error handling."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def setup_development_environment():
    """Set up complete development environment."""
    
    print("Setting up development environment for microbio-toolkit...")
    
    # Install package in development mode
    if not run_command("pip install -e .", "Installing package in development mode"):
        return False
    
    # Install development dependencies
    if not run_command("pip install -e .[dev]", "Installing development dependencies"):
        return False
    
    # Install documentation dependencies  
    if not run_command("pip install -e .[docs]", "Installing documentation dependencies"):
        return False
    
    # Set up pre-commit hooks
    if not run_command("pip install pre-commit", "Installing pre-commit"):
        return False
    
    if not run_command("pre-commit install", "Setting up pre-commit hooks"):
        return False
    
    # Run initial tests
    if not run_command("pytest tests/", "Running initial tests"):
        print("Warning: Some tests failed. Please review and fix.")
    
    # Generate initial documentation
    if not run_command("cd docs && make html", "Building documentation"):
        print("Warning: Documentation build failed.")
    
    print("\n✓ Development environment setup complete!")
    print("\nNext steps:")
    print("1. Run tests: pytest tests/")
    print("2. Check code style: flake8 microbio_toolkit/")
    print("3. Format code: black microbio_toolkit/")
    print("4. Build docs: cd docs && make html")
    print("5. Build package: python setup.py sdist bdist_wheel")

if __name__ == "__main__":
    setup_development_environment()
```

Package development enables creation of reusable, well-documented, and tested tools for the microbiology research community.
