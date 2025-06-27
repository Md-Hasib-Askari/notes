# Discrete Mathematics

## Set Theory

### Basic Concepts
- **Set**: Collection of distinct objects
- **Element**: Object belonging to a set
- **Subset**: Set contained within another set
- **Union**: A ∪ B (elements in A or B)
- **Intersection**: A ∩ B (elements in A and B)
- **Complement**: A' (elements not in A)

### Set Operations
- Commutative laws
- Associative laws
- Distributive laws
- De Morgan's laws

### Applications in ML
- Feature space representation
- Data partitioning
- Ensemble methods
- Boolean logic in decision trees

## Combinatorics

### Counting Principles
- **Addition Principle**: Sum of mutually exclusive events
- **Multiplication Principle**: Product of sequential choices
- **Inclusion-Exclusion Principle**: |A ∪ B| = |A| + |B| - |A ∩ B|

### Permutations and Combinations
- **Permutations**: P(n,r) = n!/(n-r)! (order matters)
- **Combinations**: C(n,r) = n!/(r!(n-r)!) (order doesn't matter)
- **With repetition**: Different formulas apply

### Applications
- Feature selection combinations
- Cross-validation folds
- Hyperparameter grid search
- Data augmentation strategies

## Graph Theory

### Basic Definitions
- **Graph**: G = (V, E) where V is vertices, E is edges
- **Directed vs Undirected**: Edge directionality
- **Weighted vs Unweighted**: Edge weights
- **Connected vs Disconnected**: Path existence

### Graph Properties
- **Degree**: Number of edges connected to vertex
- **Path**: Sequence of connected vertices
- **Cycle**: Path that starts and ends at same vertex
- **Tree**: Connected graph with no cycles

### Graph Algorithms
- **Breadth-First Search (BFS)**
- **Depth-First Search (DFS)**
- **Shortest Path**: Dijkstra's algorithm
- **Minimum Spanning Tree**: Kruskal's, Prim's algorithms

### Applications in ML
- **Neural Networks**: Computation graphs
- **Social Network Analysis**: User connections
- **Recommendation Systems**: Bipartite graphs
- **Knowledge Graphs**: Entity relationships
- **Clustering**: Graph-based clustering

## Logic and Boolean Algebra

### Propositional Logic
- **Propositions**: True or false statements
- **Logical Operators**: AND (∧), OR (∨), NOT (¬)
- **Implication**: A → B
- **Equivalence**: A ↔ B

### Truth Tables
- Systematic evaluation of logical expressions
- Verification of logical equivalences
- Simplification of complex expressions

### Boolean Algebra Laws
- Identity laws
- Complement laws
- Idempotent laws
- De Morgan's laws

### Applications
- **Decision Trees**: Boolean conditions
- **Feature Engineering**: Logical combinations
- **Neural Networks**: Activation functions
- **Rule-Based Systems**: Expert systems

## Practical Examples

### Example 1: Combinatorics in Cross-Validation
```python
import math
from itertools import combinations

def cv_combinations(n_samples, k_folds):
    """
    Calculate number of possible train-test splits
    """
    test_size = n_samples // k_folds
    return math.comb(n_samples, test_size)

# Example: 5-fold CV with 100 samples
n_combinations = cv_combinations(100, 5)
print(f"Possible train-test splits: {n_combinations}")
```

### Example 2: Graph-Based Feature Selection
```python
import networkx as nx
import numpy as np

def create_feature_graph(correlation_matrix, threshold=0.7):
    """
    Create graph based on feature correlations
    """
    G = nx.Graph()
    n_features = correlation_matrix.shape[0]
    
    # Add nodes
    for i in range(n_features):
        G.add_node(i)
    
    # Add edges for highly correlated features
    for i in range(n_features):
        for j in range(i+1, n_features):
            if abs(correlation_matrix[i, j]) > threshold:
                G.add_edge(i, j, weight=abs(correlation_matrix[i, j]))
    
    return G

def select_features_by_graph(G, max_features):
    """
    Select features to minimize redundancy
    """
    selected = []
    remaining = list(G.nodes())
    
    while len(selected) < max_features and remaining:
        # Select node with minimum connections to selected features
        best_node = None
        min_connections = float('inf')
        
        for node in remaining:
            connections = sum(1 for neighbor in G.neighbors(node) if neighbor in selected)
            if connections < min_connections:
                min_connections = connections
                best_node = node
        
        selected.append(best_node)
        remaining.remove(best_node)
    
    return selected
```

### Example 3: Boolean Logic in Decision Trees
```python
class BooleanRule:
    def __init__(self, feature_idx, threshold, operator='<='):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.operator = operator
    
    def evaluate(self, sample):
        value = sample[self.feature_idx]
        if self.operator == '<=':
            return value <= self.threshold
        elif self.operator == '>':
            return value > self.threshold
        elif self.operator == '==':
            return value == self.threshold
        elif self.operator == '!=':
            return value != self.threshold
    
    def __str__(self):
        return f"Feature_{self.feature_idx} {self.operator} {self.threshold}"

class BooleanExpression:
    def __init__(self, rules, operators):
        self.rules = rules  # List of BooleanRule objects
        self.operators = operators  # List of 'AND'/'OR' between rules
    
    def evaluate(self, sample):
        if not self.rules:
            return True
        
        result = self.rules[0].evaluate(sample)
        
        for i, operator in enumerate(self.operators):
            next_result = self.rules[i+1].evaluate(sample)
            if operator == 'AND':
                result = result and next_result
            elif operator == 'OR':
                result = result or next_result
        
        return result
```

### Example 4: Set Operations for Data Processing
```python
def analyze_feature_sets(dataset1_features, dataset2_features):
    """
    Analyze overlap between feature sets
    """
    set1 = set(dataset1_features)
    set2 = set(dataset2_features)
    
    analysis = {
        'dataset1_only': set1 - set2,
        'dataset2_only': set2 - set1,
        'common_features': set1 & set2,
        'all_features': set1 | set2,
        'jaccard_similarity': len(set1 & set2) / len(set1 | set2)
    }
    
    return analysis

# Example usage
features_train = ['age', 'income', 'education', 'experience']
features_test = ['age', 'income', 'location', 'experience']

analysis = analyze_feature_sets(features_train, features_test)
print(f"Common features: {analysis['common_features']}")
print(f"Jaccard similarity: {analysis['jaccard_similarity']:.2f}")
```

## Learning Objectives
- [ ] Master set operations and logic
- [ ] Apply combinatorics to ML problems
- [ ] Understand graph theory basics
- [ ] Use boolean algebra in algorithms
- [ ] Implement discrete math concepts in code

## Applications Summary

### Set Theory
- Data preprocessing and cleaning
- Feature space analysis
- Ensemble method design

### Combinatorics
- Cross-validation strategies
- Hyperparameter optimization
- Feature selection methods

### Graph Theory
- Neural network architectures
- Social network analysis
- Knowledge representation

### Boolean Logic
- Decision tree construction
- Rule-based systems
- Feature engineering

## Resources
- "Discrete Mathematics and Its Applications" by Rosen
- NetworkX documentation for graph algorithms
- Combinatorics libraries in Python
- Logic programming resources
