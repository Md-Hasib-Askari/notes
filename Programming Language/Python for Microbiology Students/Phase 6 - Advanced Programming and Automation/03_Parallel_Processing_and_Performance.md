# Parallel Processing and Performance

## Multiprocessing for Large Datasets
Use multiprocessing to accelerate computationally intensive microbiology tasks.

```python
import multiprocessing as mp
import pandas as pd
import numpy as np
from functools import partial

def analyze_single_genome(genome_file):
    """Analyze a single genome file"""
    from Bio import SeqIO
    
    # Simulate genome analysis
    sequences = list(SeqIO.parse(genome_file, "fasta"))
    
    results = {
        'filename': genome_file,
        'num_contigs': len(sequences),
        'total_length': sum(len(seq) for seq in sequences),
        'gc_content': calculate_gc_content([str(seq.seq) for seq in sequences])
    }
    
    return results

def calculate_gc_content(sequences):
    """Calculate GC content for sequences"""
    total_bases = 0
    gc_bases = 0
    
    for seq in sequences:
        total_bases += len(seq)
        gc_bases += seq.upper().count('G') + seq.upper().count('C')
    
    return (gc_bases / total_bases * 100) if total_bases > 0 else 0

def parallel_genome_analysis(genome_files, num_processes=None):
    """Analyze multiple genomes in parallel"""
    
    if num_processes is None:
        num_processes = mp.cpu_count() - 1
    
    print(f"Analyzing {len(genome_files)} genomes using {num_processes} processes...")
    
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(analyze_single_genome, genome_files)
    
    return pd.DataFrame(results)

# Process microbiome samples in parallel
def process_microbiome_sample(sample_data):
    """Process single microbiome sample"""
    sample_id, abundance_data = sample_data
    
    # Calculate diversity metrics
    shannon_diversity = -sum(p * np.log(p) for p in abundance_data if p > 0)
    simpson_diversity = 1 - sum(p**2 for p in abundance_data)
    observed_species = sum(1 for p in abundance_data if p > 0)
    
    return {
        'sample_id': sample_id,
        'shannon': shannon_diversity,
        'simpson': simpson_diversity,
        'observed_species': observed_species
    }

def parallel_microbiome_analysis(abundance_matrix):
    """Analyze microbiome diversity in parallel"""
    
    # Prepare data for parallel processing
    sample_data = [(idx, row.values) for idx, row in abundance_matrix.iterrows()]
    
    with mp.Pool() as pool:
        diversity_results = pool.map(process_microbiome_sample, sample_data)
    
    return pd.DataFrame(diversity_results)

# Example usage
# genome_files = ['genome1.fasta', 'genome2.fasta', 'genome3.fasta']
# results = parallel_genome_analysis(genome_files)
```

## Optimizing Computational Bottlenecks
Identify and optimize performance bottlenecks in biological data analysis.

```python
import time
import cProfile
import pstats
from memory_profiler import profile
import numpy as np

def profile_function(func):
    """Decorator to profile function performance"""
    def wrapper(*args, **kwargs):
        # Time profiling
        start_time = time.time()
        
        # Memory and CPU profiling
        pr = cProfile.Profile()
        pr.enable()
        
        result = func(*args, **kwargs)
        
        pr.disable()
        end_time = time.time()
        
        # Print timing results
        print(f"\n{func.__name__} completed in {end_time - start_time:.2f} seconds")
        
        # Print CPU profiling results
        stats = pstats.Stats(pr)
        stats.sort_stats('cumulative')
        stats.print_stats(10)  # Top 10 functions
        
        return result
    
    return wrapper

# Slow vs optimized sequence analysis
def slow_sequence_analysis(sequences):
    """Inefficient sequence analysis"""
    gc_contents = []
    
    for seq in sequences:
        gc_count = 0
        total_count = 0
        
        # Inefficient character-by-character counting
        for nucleotide in seq:
            total_count += 1
            if nucleotide.upper() in ['G', 'C']:
                gc_count += 1
        
        gc_content = (gc_count / total_count * 100) if total_count > 0 else 0
        gc_contents.append(gc_content)
    
    return gc_contents

@profile_function
def optimized_sequence_analysis(sequences):
    """Optimized sequence analysis using vectorization"""
    gc_contents = []
    
    for seq in sequences:
        # Use string methods for counting
        seq_upper = seq.upper()
        gc_count = seq_upper.count('G') + seq_upper.count('C')
        total_count = len(seq)
        
        gc_content = (gc_count / total_count * 100) if total_count > 0 else 0
        gc_contents.append(gc_content)
    
    return gc_contents

def vectorized_sequence_analysis(sequences):
    """Fully vectorized analysis using NumPy"""
    import numpy as np
    
    # Convert to NumPy arrays for vectorization
    sequences_array = np.array([list(seq.upper()) for seq in sequences])
    
    # Vectorized GC counting
    gc_mask = (sequences_array == 'G') | (sequences_array == 'C')
    gc_counts = np.sum(gc_mask, axis=1)
    total_counts = np.array([len(seq) for seq in sequences])
    
    gc_contents = (gc_counts / total_counts * 100)
    
    return gc_contents

# Benchmark different approaches
def benchmark_sequence_analysis():
    """Benchmark different sequence analysis methods"""
    
    # Generate test data
    np.random.seed(42)
    sequences = []
    for _ in range(1000):
        length = np.random.randint(100, 1000)
        seq = ''.join(np.random.choice(['A', 'T', 'G', 'C'], length))
        sequences.append(seq)
    
    print("Benchmarking sequence analysis methods...")
    
    # Test slow method
    start = time.time()
    slow_results = slow_sequence_analysis(sequences[:100])  # Smaller subset
    slow_time = time.time() - start
    print(f"Slow method: {slow_time:.3f} seconds")
    
    # Test optimized method
    start = time.time()
    opt_results = optimized_sequence_analysis(sequences)
    opt_time = time.time() - start
    print(f"Optimized method: {opt_time:.3f} seconds")
    
    # Test vectorized method
    start = time.time()
    vec_results = vectorized_sequence_analysis(sequences)
    vec_time = time.time() - start
    print(f"Vectorized method: {vec_time:.3f} seconds")
    
    print(f"Speedup (opt vs slow): {slow_time/opt_time:.1f}x")
    print(f"Speedup (vec vs slow): {slow_time/vec_time:.1f}x")

# benchmark_sequence_analysis()
```

## Memory Management for Big Data
Efficient memory management techniques for large biological datasets.

```python
import gc
import psutil
import pandas as pd

def monitor_memory_usage():
    """Monitor current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        'percent': process.memory_percent()
    }

def process_large_dataset_chunked(file_path, chunk_size=10000):
    """Process large datasets in chunks to manage memory"""
    
    print("Processing large dataset in chunks...")
    memory_start = monitor_memory_usage()
    print(f"Initial memory: {memory_start['rss_mb']:.1f} MB")
    
    results = []
    
    # Read file in chunks
    for chunk_num, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
        
        # Process chunk
        processed_chunk = chunk.groupby('sample_id').agg({
            'abundance': ['mean', 'std', 'count']
        }).reset_index()
        
        # Store only aggregated results
        results.append(processed_chunk)
        
        # Force garbage collection periodically
        if chunk_num % 10 == 0:
            gc.collect()
            memory_current = monitor_memory_usage()
            print(f"Chunk {chunk_num}: {memory_current['rss_mb']:.1f} MB")
        
        # Memory limit check
        if memory_current['rss_mb'] > 4000:  # 4GB limit
            print("Memory limit reached, forcing cleanup...")
            gc.collect()
    
    # Combine results
    final_result = pd.concat(results, ignore_index=True)
    
    memory_end = monitor_memory_usage()
    print(f"Final memory: {memory_end['rss_mb']:.1f} MB")
    
    return final_result

class MemoryEfficientAnalysis:
    """Class for memory-efficient biological data analysis"""
    
    def __init__(self, max_memory_mb=2000):
        self.max_memory_mb = max_memory_mb
        self.temp_results = []
    
    def add_sample_result(self, result):
        """Add sample result with memory monitoring"""
        self.temp_results.append(result)
        
        # Check memory usage
        current_memory = monitor_memory_usage()['rss_mb']
        
        if current_memory > self.max_memory_mb:
            print(f"Memory usage ({current_memory:.1f} MB) exceeds limit, consolidating...")
            self._consolidate_results()
    
    def _consolidate_results(self):
        """Consolidate temporary results to save memory"""
        if self.temp_results:
            # Combine and summarize results
            combined = pd.DataFrame(self.temp_results)
            summary = combined.groupby('sample_type').agg({
                'diversity': ['mean', 'std'],
                'richness': ['mean', 'std']
            })
            
            # Clear temporary results and store summary
            self.temp_results = [summary.to_dict()]
            
            # Force garbage collection
            gc.collect()
            
            print(f"Consolidated {len(combined)} results into summary")

# Context manager for memory monitoring
class MemoryMonitor:
    """Context manager to monitor memory usage"""
    
    def __init__(self, operation_name):
        self.operation_name = operation_name
        self.start_memory = None
    
    def __enter__(self):
        self.start_memory = monitor_memory_usage()
        print(f"Starting {self.operation_name}...")
        print(f"Initial memory: {self.start_memory['rss_mb']:.1f} MB")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_memory = monitor_memory_usage()
        memory_diff = end_memory['rss_mb'] - self.start_memory['rss_mb']
        
        print(f"Completed {self.operation_name}")
        print(f"Final memory: {end_memory['rss_mb']:.1f} MB")
        print(f"Memory change: {memory_diff:+.1f} MB")

# Usage example
with MemoryMonitor("Large dataset analysis"):
    analyzer = MemoryEfficientAnalysis(max_memory_mb=1000)
    
    # Simulate processing many samples
    for i in range(100):
        sample_result = {
            'sample_id': f'S{i:03d}',
            'sample_type': 'gut' if i % 2 == 0 else 'oral',
            'diversity': np.random.normal(3.5, 0.5),
            'richness': np.random.randint(50, 200)
        }
        analyzer.add_sample_result(sample_result)
```

## Profiling and Benchmarking Code
Comprehensive profiling tools for optimizing biological data analysis code.

```python
import timeit
import line_profiler
import memory_profiler

def time_function(func, *args, number=1000):
    """Time function execution"""
    
    timer = timeit.Timer(lambda: func(*args))
    execution_time = timer.timeit(number=number)
    
    print(f"{func.__name__}: {execution_time/number*1000:.3f} ms per call")
    return execution_time

def compare_implementations(*implementations, test_data=None):
    """Compare multiple function implementations"""
    
    if test_data is None:
        # Generate test data
        test_data = [''.join(np.random.choice(['A', 'T', 'G', 'C'], 1000)) for _ in range(100)]
    
    results = {}
    
    for impl in implementations:
        try:
            # Time the implementation
            start_time = time.time()
            result = impl(test_data)
            end_time = time.time()
            
            execution_time = end_time - start_time
            results[impl.__name__] = {
                'time': execution_time,
                'result_length': len(result) if hasattr(result, '__len__') else 1
            }
            
            print(f"{impl.__name__}: {execution_time:.4f} seconds")
            
        except Exception as e:
            print(f"{impl.__name__} failed: {e}")
            results[impl.__name__] = {'time': float('inf'), 'error': str(e)}
    
    # Find fastest implementation
    fastest = min(results.keys(), key=lambda k: results[k]['time'])
    print(f"\nFastest implementation: {fastest}")
    
    return results

# Memory profiling decorator
def memory_profile(func):
    """Decorator for memory profiling"""
    def wrapper(*args, **kwargs):
        from memory_profiler import profile
        return profile(func)(*args, **kwargs)
    return wrapper

@memory_profile
def memory_intensive_analysis(large_dataset):
    """Example memory-intensive function"""
    
    # Simulate memory-intensive operations
    intermediate_results = []
    
    for chunk in large_dataset:
        processed = [x**2 for x in chunk]  # Memory-intensive operation
        intermediate_results.append(processed)
    
    final_result = sum(sum(chunk) for chunk in intermediate_results)
    return final_result

# Performance testing suite
class PerformanceTestSuite:
    """Comprehensive performance testing for biological analysis functions"""
    
    def __init__(self):
        self.test_results = {}
    
    def add_test(self, test_name, func, test_data, expected_result=None):
        """Add a performance test"""
        
        print(f"\nTesting {test_name}...")
        
        # Correctness test
        try:
            result = func(test_data)
            correct = True
            
            if expected_result is not None:
                if isinstance(expected_result, (int, float)):
                    correct = abs(result - expected_result) < 1e-6
                else:
                    correct = result == expected_result
            
        except Exception as e:
            print(f"Function failed: {e}")
            correct = False
            result = None
        
        # Performance test
        if correct:
            execution_times = []
            for _ in range(5):  # Run 5 times
                start = time.time()
                func(test_data)
                end = time.time()
                execution_times.append(end - start)
            
            avg_time = np.mean(execution_times)
            std_time = np.std(execution_times)
            
            self.test_results[test_name] = {
                'correct': correct,
                'avg_time': avg_time,
                'std_time': std_time,
                'result': result
            }
            
            print(f"Average time: {avg_time:.4f} ± {std_time:.4f} seconds")
        
        else:
            self.test_results[test_name] = {
                'correct': False,
                'avg_time': float('inf'),
                'std_time': 0,
                'result': None
            }
    
    def print_summary(self):
        """Print performance test summary"""
        
        print("\n" + "="*50)
        print("PERFORMANCE TEST SUMMARY")
        print("="*50)
        
        for test_name, results in self.test_results.items():
            status = "PASS" if results['correct'] else "FAIL"
            time_str = f"{results['avg_time']:.4f}s" if results['correct'] else "N/A"
            
            print(f"{test_name:30} {status:6} {time_str:>10}")

# Example usage
# suite = PerformanceTestSuite()
# suite.add_test("GC Content Slow", slow_sequence_analysis, test_sequences)
# suite.add_test("GC Content Optimized", optimized_sequence_analysis, test_sequences)
# suite.add_test("GC Content Vectorized", vectorized_sequence_analysis, test_sequences)
# suite.print_summary()
```

These performance optimization techniques enable efficient processing of large biological datasets and computational workflows.
