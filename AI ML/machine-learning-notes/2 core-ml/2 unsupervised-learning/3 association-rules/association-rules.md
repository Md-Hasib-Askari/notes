# Association Rules

## Overview
Association rules discover relationships between different items in datasets. Primarily used in market basket analysis to find items frequently bought together.

**Key Concept**: If customers buy X, they also tend to buy Y with certain probability.

## Core Concepts

### 1. Market Basket Analysis
Analysis of shopping baskets to identify purchasing patterns.

```python
# Example transaction data
transactions = [
    ['bread', 'milk', 'butter'],
    ['bread', 'milk'],
    ['milk', 'butter', 'cheese'],
    ['bread', 'butter'],
    ['milk', 'cheese']
]
```

### 2. Key Metrics

#### Support
Frequency of itemset in dataset
```
Support(X) = Count(X) / Total_transactions
```

#### Confidence
Probability of Y given X
```
Confidence(X → Y) = Support(X ∪ Y) / Support(X)
```

#### Lift
How much more likely Y is when X is present
```
Lift(X → Y) = Confidence(X → Y) / Support(Y)
```

#### Conviction
Measures dependency (higher = stronger rule)
```
Conviction(X → Y) = (1 - Support(Y)) / (1 - Confidence(X → Y))
```

## Algorithms

### 1. Apriori Algorithm
Most common algorithm for mining frequent itemsets.

```python
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

# Sample data
transactions = [
    ['milk', 'eggs', 'bread', 'cheese'],
    ['eggs', 'bread'],
    ['milk', 'bread'],
    ['eggs', 'bread', 'butter'],
    ['milk', 'eggs', 'bread', 'butter']
]

# Transform to binary matrix
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

print("Transaction Matrix:")
print(df)

# Find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
```

### 2. FP-Growth Algorithm
More efficient than Apriori for large datasets.

```python
from mlxtend.frequent_patterns import fpgrowth

# Using FP-Growth
frequent_itemsets_fp = fpgrowth(df, min_support=0.4, use_colnames=True)
print("FP-Growth Results:")
print(frequent_itemsets_fp)
```

### 3. Eclat Algorithm
Uses vertical data representation.

```python
# Eclat implementation (simplified)
def eclat(transactions, min_support):
    # Convert to vertical representation
    vertical_db = {}
    for tid, transaction in enumerate(transactions):
        for item in transaction:
            if item not in vertical_db:
                vertical_db[item] = set()
            vertical_db[item].add(tid)
    
    # Find frequent 1-itemsets
    frequent_items = {}
    total_transactions = len(transactions)
    
    for item, tids in vertical_db.items():
        support = len(tids) / total_transactions
        if support >= min_support:
            frequent_items[frozenset([item])] = tids
    
    return frequent_items

# Example usage
frequent = eclat(transactions, min_support=0.4)
print("Eclat Results:")
for itemset, tids in frequent.items():
    print(f"{set(itemset)}: Support = {len(tids)/len(transactions):.2f}")
```

## Practical Implementation

### Complete Market Basket Analysis

```python
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample retail data
np.random.seed(42)
products = ['bread', 'milk', 'eggs', 'cheese', 'butter', 'yogurt', 'apple', 'banana']
transactions = []

for _ in range(1000):
    # Random basket size
    basket_size = np.random.randint(2, 6)
    basket = np.random.choice(products, basket_size, replace=False).tolist()
    
    # Add some patterns
    if 'bread' in basket and np.random.random() > 0.3:
        if 'butter' not in basket:
            basket.append('butter')
    if 'milk' in basket and np.random.random() > 0.4:
        if 'eggs' not in basket:
            basket.append('eggs')
    
    transactions.append(basket)

print(f"Generated {len(transactions)} transactions")
print("Sample transactions:", transactions[:5])

# Convert to binary matrix
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
basket_df = pd.DataFrame(te_ary, columns=te.columns_)

# Analyze frequent itemsets
frequent_itemsets = apriori(basket_df, min_support=0.05, use_colnames=True)
print(f"\nFound {len(frequent_itemsets)} frequent itemsets")

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
rules = rules.sort_values('lift', ascending=False)

print("\nTop 10 Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# Visualize results
plt.figure(figsize=(12, 8))

# Support vs Confidence scatter plot
plt.subplot(2, 2, 1)
plt.scatter(rules['support'], rules['confidence'], alpha=0.7, c=rules['lift'], cmap='viridis')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs Confidence (colored by Lift)')
plt.colorbar(label='Lift')

# Lift distribution
plt.subplot(2, 2, 2)
plt.hist(rules['lift'], bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('Lift')
plt.ylabel('Frequency')
plt.title('Distribution of Lift Values')

# Top rules by lift
plt.subplot(2, 2, 3)
top_rules = rules.head(10)
rule_labels = [f"{list(ant)} → {list(cons)}" for ant, cons in zip(top_rules['antecedents'], top_rules['consequents'])]
plt.barh(range(len(top_rules)), top_rules['lift'])
plt.yticks(range(len(top_rules)), [f"Rule {i+1}" for i in range(len(top_rules))])
plt.xlabel('Lift')
plt.title('Top 10 Rules by Lift')

# Confidence vs Lift
plt.subplot(2, 2, 4)
plt.scatter(rules['confidence'], rules['lift'], alpha=0.7)
plt.xlabel('Confidence')
plt.ylabel('Lift')
plt.title('Confidence vs Lift')

plt.tight_layout()
plt.show()
```

### Rule Interpretation

```python
def interpret_rules(rules, top_n=5):
    """Interpret association rules in business terms"""
    print(f"Top {top_n} Association Rules Analysis:")
    print("=" * 50)
    
    for idx, rule in rules.head(top_n).iterrows():
        antecedent = list(rule['antecedents'])[0] if len(rule['antecedents']) == 1 else list(rule['antecedents'])
        consequent = list(rule['consequents'])[0] if len(rule['consequents']) == 1 else list(rule['consequents'])
        
        print(f"\nRule {idx + 1}: {antecedent} → {consequent}")
        print(f"Support: {rule['support']:.3f} ({rule['support']*100:.1f}% of transactions)")
        print(f"Confidence: {rule['confidence']:.3f} ({rule['confidence']*100:.1f}% probability)")
        print(f"Lift: {rule['lift']:.3f}", end="")
        
        if rule['lift'] > 1:
            print(" (Positive correlation - items bought together more than expected)")
        elif rule['lift'] < 1:
            print(" (Negative correlation - items bought together less than expected)")
        else:
            print(" (No correlation)")
            
        print(f"Conviction: {rule['conviction']:.3f}")
        
        # Business interpretation
        if rule['confidence'] > 0.8 and rule['lift'] > 1.2:
            print("💡 Strong rule - Consider bundling these items!")
        elif rule['confidence'] > 0.6 and rule['lift'] > 1.1:
            print("💡 Moderate rule - Good for cross-selling recommendations")

# Apply interpretation
interpret_rules(rules)
```

## Applications

### 1. E-commerce Recommendations

```python
class RecommendationEngine:
    def __init__(self, rules):
        self.rules = rules
        
    def get_recommendations(self, basket, top_n=3):
        """Get product recommendations based on current basket"""
        recommendations = []
        
        for _, rule in self.rules.iterrows():
            antecedent = set(rule['antecedents'])
            consequent = set(rule['consequents'])
            
            # Check if basket contains antecedent
            if antecedent.issubset(set(basket)):
                # Recommend consequent if not already in basket
                if not consequent.issubset(set(basket)):
                    recommendations.append({
                        'items': list(consequent),
                        'confidence': rule['confidence'],
                        'lift': rule['lift']
                    })
        
        # Sort by confidence and return top_n
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        return recommendations[:top_n]

# Example usage
recommender = RecommendationEngine(rules)
current_basket = ['bread', 'milk']
suggestions = recommender.get_recommendations(current_basket)

print(f"Current basket: {current_basket}")
print("Recommendations:")
for i, rec in enumerate(suggestions, 1):
    print(f"{i}. {rec['items']} (confidence: {rec['confidence']:.3f}, lift: {rec['lift']:.3f})")
```

### 2. Web Usage Pattern Analysis

```python
# Web clickstream analysis
web_sessions = [
    ['homepage', 'products', 'cart'],
    ['homepage', 'about', 'contact'],
    ['products', 'details', 'cart', 'checkout'],
    ['homepage', 'products', 'details'],
    ['search', 'products', 'cart']
]

# Convert to transaction format
te_web = TransactionEncoder()
te_web_ary = te_web.fit(web_sessions).transform(web_sessions)
web_df = pd.DataFrame(te_web_ary, columns=te_web.columns_)

# Find navigation patterns
web_frequent = apriori(web_df, min_support=0.3, use_colnames=True)
web_rules = association_rules(web_frequent, metric="confidence", min_threshold=0.5)

print("Web Navigation Patterns:")
for _, rule in web_rules.iterrows():
    print(f"After visiting {list(rule['antecedents'])}, "
          f"users go to {list(rule['consequents'])} "
          f"with {rule['confidence']:.1%} probability")
```

### 3. Cross-selling Strategy

```python
def generate_cross_sell_strategy(rules, revenue_data=None):
    """Generate cross-selling strategies based on association rules"""
    
    strategies = []
    
    for _, rule in rules.iterrows():
        if rule['confidence'] > 0.5 and rule['lift'] > 1.2:
            strategy = {
                'trigger_items': list(rule['antecedents']),
                'recommend_items': list(rule['consequents']),
                'confidence': rule['confidence'],
                'lift': rule['lift'],
                'expected_increase': (rule['lift'] - 1) * 100
            }
            strategies.append(strategy)
    
    return sorted(strategies, key=lambda x: x['expected_increase'], reverse=True)

# Generate strategies
strategies = generate_cross_sell_strategy(rules)

print("Cross-selling Strategies:")
print("=" * 40)
for i, strategy in enumerate(strategies[:5], 1):
    print(f"\nStrategy {i}:")
    print(f"When customer buys: {strategy['trigger_items']}")
    print(f"Recommend: {strategy['recommend_items']}")
    print(f"Success rate: {strategy['confidence']:.1%}")
    print(f"Expected sales increase: {strategy['expected_increase']:.1f}%")
```

## Model Evaluation

### Performance Metrics

```python
def evaluate_association_rules(rules):
    """Evaluate quality of association rules"""
    
    metrics = {
        'total_rules': len(rules),
        'avg_confidence': rules['confidence'].mean(),
        'avg_lift': rules['lift'].mean(),
        'avg_support': rules['support'].mean(),
        'strong_rules': len(rules[(rules['confidence'] > 0.8) & (rules['lift'] > 1.5)]),
        'actionable_rules': len(rules[(rules['confidence'] > 0.6) & (rules['lift'] > 1.2)])
    }
    
    print("Association Rules Evaluation:")
    print("=" * 30)
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.3f}")
        else:
            print(f"{metric}: {value}")
    
    return metrics

# Evaluate rules
evaluation = evaluate_association_rules(rules)
```

## Best Practices

### 1. Parameter Tuning
```python
# Test different support and confidence thresholds
support_values = [0.01, 0.05, 0.1, 0.2]
confidence_values = [0.3, 0.5, 0.7, 0.9]

results = []
for sup in support_values:
    for conf in confidence_values:
        freq_items = apriori(basket_df, min_support=sup, use_colnames=True)
        if len(freq_items) > 0:
            rules_test = association_rules(freq_items, metric="confidence", min_threshold=conf)
            results.append({
                'support': sup,
                'confidence': conf,
                'num_rules': len(rules_test),
                'avg_lift': rules_test['lift'].mean() if len(rules_test) > 0 else 0
            })

# Find optimal parameters
results_df = pd.DataFrame(results)
print("Parameter Tuning Results:")
print(results_df.sort_values('avg_lift', ascending=False).head())
```

### 2. Rule Filtering
```python
def filter_meaningful_rules(rules, min_lift=1.1, min_confidence=0.5, min_support=0.01):
    """Filter rules based on business criteria"""
    
    filtered = rules[
        (rules['lift'] >= min_lift) &
        (rules['confidence'] >= min_confidence) &
        (rules['support'] >= min_support)
    ]
    
    # Remove redundant rules (subset rules with lower confidence)
    unique_rules = []
    for _, rule in filtered.iterrows():
        is_redundant = False
        for _, other_rule in filtered.iterrows():
            if (rule['antecedents'].issubset(other_rule['antecedents']) and
                rule['consequents'] == other_rule['consequents'] and
                rule['confidence'] < other_rule['confidence']):
                is_redundant = True
                break
        if not is_redundant:
            unique_rules.append(rule)
    
    return pd.DataFrame(unique_rules)

# Apply filtering
meaningful_rules = filter_meaningful_rules(rules)
print(f"Filtered from {len(rules)} to {len(meaningful_rules)} meaningful rules")
```

## Common Pitfalls

1. **Low Support Threshold**: May generate too many rules
2. **High Support Threshold**: May miss rare but valuable patterns
3. **Ignoring Lift**: High confidence doesn't mean strong association
4. **Overfitting**: Rules that don't generalize to new data
5. **Seasonal Effects**: Patterns may change over time

## Key Takeaways

- Association rules find "if-then" relationships in data
- Support measures frequency, confidence measures reliability
- Lift > 1 indicates positive correlation
- Apriori is most common but FP-Growth is more efficient
- Applications include recommendations, cross-selling, and pattern discovery
- Always validate rules with business knowledge
- Consider temporal and seasonal effects in real applications