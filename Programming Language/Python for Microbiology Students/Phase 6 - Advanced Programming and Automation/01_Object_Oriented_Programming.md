# Object-Oriented Programming

## Classes and Objects for Biological Entities
Create reusable classes to represent biological entities and processes.

```python
class BacterialStrain:
    """Class to represent a bacterial strain"""
    
    def __init__(self, name, species, gram_status, growth_temp=37):
        self.name = name
        self.species = species
        self.gram_status = gram_status
        self.growth_temp = growth_temp
        self.growth_data = []
    
    def add_growth_measurement(self, time, od600):
        """Add growth measurement"""
        self.growth_data.append({'time': time, 'od600': od600})
    
    def calculate_growth_rate(self):
        """Calculate exponential growth rate"""
        import numpy as np
        
        if len(self.growth_data) < 2:
            return None
        
        times = [d['time'] for d in self.growth_data]
        ods = [d['od600'] for d in self.growth_data]
        
        # Use log-linear regression
        log_ods = np.log(ods)
        growth_rate = np.polyfit(times, log_ods, 1)[0]
        
        return growth_rate
    
    def __str__(self):
        return f"{self.name} ({self.species}) - {self.gram_status}"

# Usage example
ecoli = BacterialStrain("DH5α", "Escherichia coli", "Gram-negative")
ecoli.add_growth_measurement(0, 0.1)
ecoli.add_growth_measurement(2, 0.3)
ecoli.add_growth_measurement(4, 0.8)
print(f"Growth rate: {ecoli.calculate_growth_rate():.3f} h⁻¹")
```

## Inheritance and Polymorphism
Use inheritance to create specialized biological classes.

```python
class Microorganism:
    """Base class for all microorganisms"""
    
    def __init__(self, name, domain):
        self.name = name
        self.domain = domain
    
    def describe(self):
        return f"{self.name} belongs to {self.domain}"

class Bacterium(Microorganism):
    """Specialized class for bacteria"""
    
    def __init__(self, name, gram_status, shape):
        super().__init__(name, "Bacteria")
        self.gram_status = gram_status
        self.shape = shape
    
    def describe(self):
        base = super().describe()
        return f"{base}. {self.gram_status}, {self.shape}-shaped"

class Virus(Microorganism):
    """Specialized class for viruses"""
    
    def __init__(self, name, host, genome_type):
        super().__init__(name, "Virus")
        self.host = host
        self.genome_type = genome_type
    
    def describe(self):
        base = super().describe()
        return f"{base}. Infects {self.host}, {self.genome_type} genome"

# Polymorphism example
organisms = [
    Bacterium("E. coli", "Gram-negative", "rod"),
    Virus("Influenza A", "humans", "RNA")
]

for organism in organisms:
    print(organism.describe())  # Each calls appropriate describe method
```

## Designing Reusable Code for Lab Workflows
Create modular code for common laboratory workflows.

```python
class LabExperiment:
    """Base class for laboratory experiments"""
    
    def __init__(self, experiment_id, researcher):
        self.experiment_id = experiment_id
        self.researcher = researcher
        self.start_time = None
        self.end_time = None
        self.results = {}
    
    def start_experiment(self):
        """Start the experiment"""
        from datetime import datetime
        self.start_time = datetime.now()
        print(f"Experiment {self.experiment_id} started by {self.researcher}")
    
    def end_experiment(self):
        """End the experiment"""
        from datetime import datetime
        self.end_time = datetime.now()
        duration = self.end_time - self.start_time
        print(f"Experiment completed in {duration}")
    
    def save_results(self, filename):
        """Save experiment results"""
        import json
        
        data = {
            'experiment_id': self.experiment_id,
            'researcher': self.researcher,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

class GrowthCurveExperiment(LabExperiment):
    """Specialized experiment for growth curves"""
    
    def __init__(self, experiment_id, researcher, strains):
        super().__init__(experiment_id, researcher)
        self.strains = strains
        self.measurements = {strain.name: [] for strain in strains}
    
    def record_measurement(self, strain_name, time, od600):
        """Record growth measurement"""
        if strain_name in self.measurements:
            self.measurements[strain_name].append({'time': time, 'od600': od600})
    
    def analyze_results(self):
        """Analyze growth curve results"""
        for strain_name, data in self.measurements.items():
            if len(data) >= 3:
                # Calculate doubling time
                max_rate_idx = self._find_exponential_phase(data)
                if max_rate_idx:
                    doubling_time = self._calculate_doubling_time(data, max_rate_idx)
                    self.results[strain_name] = {'doubling_time': doubling_time}
    
    def _find_exponential_phase(self, data):
        # Simplified: find steepest growth phase
        max_rate = 0
        max_idx = 0
        for i in range(1, len(data)):
            rate = (data[i]['od600'] - data[i-1]['od600']) / (data[i]['time'] - data[i-1]['time'])
            if rate > max_rate:
                max_rate = rate
                max_idx = i
        return max_idx
    
    def _calculate_doubling_time(self, data, idx):
        # Simplified doubling time calculation
        if idx > 0:
            rate = (data[idx]['od600'] - data[idx-1]['od600']) / (data[idx]['time'] - data[idx-1]['time'])
            return 0.693 / rate if rate > 0 else None
        return None

# Usage
strains = [ecoli]
experiment = GrowthCurveExperiment("GC001", "Dr. Smith", strains)
experiment.start_experiment()
experiment.record_measurement("DH5α", 0, 0.1)
experiment.record_measurement("DH5α", 2, 0.3)
experiment.analyze_results()
experiment.end_experiment()
```

## Exception Handling and Debugging
Implement robust error handling for laboratory data processing.

```python
class DataProcessingError(Exception):
    """Custom exception for data processing errors"""
    pass

def safe_file_processor(file_path, expected_format="csv"):
    """Safely process laboratory data files"""
    
    try:
        import pandas as pd
        
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Validate file format
        if not file_path.endswith(f".{expected_format}"):
            raise ValueError(f"Expected {expected_format} file, got {file_path}")
        
        # Try to read file
        if expected_format == "csv":
            data = pd.read_csv(file_path)
        else:
            raise DataProcessingError(f"Unsupported format: {expected_format}")
        
        # Validate data structure
        if data.empty:
            raise DataProcessingError("File contains no data")
        
        return data
        
    except FileNotFoundError as e:
        print(f"File error: {e}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: CSV file is empty")
        return None
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV: {e}")
        return None
    except DataProcessingError as e:
        print(f"Data processing error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Debugging helper
def debug_growth_data(data):
    """Debug growth curve data"""
    
    print("=== Growth Data Debug ===")
    print(f"Data shape: {data.shape if hasattr(data, 'shape') else 'Not a DataFrame'}")
    
    if hasattr(data, 'columns'):
        print(f"Columns: {list(data.columns)}")
        
        # Check for negative values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            negative_count = (data[col] < 0).sum()
            if negative_count > 0:
                print(f"Warning: {negative_count} negative values in {col}")
        
        # Check for missing values
        missing = data.isnull().sum()
        if missing.any():
            print(f"Missing values: {missing[missing > 0].to_dict()}")

# Usage with error handling
try:
    lab_data = safe_file_processor("growth_data.csv")
    if lab_data is not None:
        debug_growth_data(lab_data)
except Exception as e:
    print(f"Failed to process data: {e}")
```

Object-oriented programming enables creation of maintainable, reusable code for complex laboratory workflows and biological data analysis.
