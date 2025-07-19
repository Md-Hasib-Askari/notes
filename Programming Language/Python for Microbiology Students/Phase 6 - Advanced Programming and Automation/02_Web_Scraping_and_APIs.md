# Web Scraping and APIs

## Extracting Data from Biological Databases
Automate data collection from public biological databases using web scraping.

```python
import requests
from bs4 import BeautifulSoup
import time
import pandas as pd

def scrape_uniprot_protein_info(protein_ids):
    """Scrape protein information from UniProt"""
    
    protein_data = []
    base_url = "https://www.uniprot.org/uniprot/"
    
    for protein_id in protein_ids:
        try:
            url = f"{base_url}{protein_id}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract protein name
            name_element = soup.find('h1')
            protein_name = name_element.text.strip() if name_element else "Unknown"
            
            # Extract organism
            organism_element = soup.find('a', {'data-testid': 'organism-name'})
            organism = organism_element.text.strip() if organism_element else "Unknown"
            
            # Extract function
            function_section = soup.find('section', {'data-testid': 'function'})
            function = function_section.get_text(strip=True) if function_section else "Not available"
            
            protein_data.append({
                'protein_id': protein_id,
                'name': protein_name,
                'organism': organism,
                'function': function[:200] + "..." if len(function) > 200 else function
            })
            
            # Rate limiting
            time.sleep(1)
            
        except requests.RequestException as e:
            print(f"Error fetching {protein_id}: {e}")
            protein_data.append({
                'protein_id': protein_id,
                'name': 'Error',
                'organism': 'Error',
                'function': 'Error'
            })
    
    return pd.DataFrame(protein_data)

# Example usage
protein_ids = ['P0A7G6', 'P0A6F5', 'P0A9P0']
protein_info = scrape_uniprot_protein_info(protein_ids)
print(protein_info)
```

## Automated Literature Searches
Automate PubMed searches for research literature.

```python
from Bio import Entrez
import xml.etree.ElementTree as ET

def search_pubmed_literature(query, max_results=10, email="your.email@university.edu"):
    """Search PubMed for literature"""
    
    Entrez.email = email
    
    try:
        # Search for articles
        search_handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        search_results = Entrez.read(search_handle)
        search_handle.close()
        
        id_list = search_results["IdList"]
        
        if not id_list:
            return pd.DataFrame()
        
        # Fetch article details
        fetch_handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="xml")
        articles = Entrez.read(fetch_handle)
        fetch_handle.close()
        
        # Parse article information
        literature_data = []
        
        for article in articles['PubmedArticle']:
            try:
                medline_citation = article['MedlineCitation']
                
                # Extract title
                title = medline_citation['Article']['ArticleTitle']
                
                # Extract authors
                author_list = medline_citation['Article'].get('AuthorList', [])
                authors = []
                for author in author_list[:3]:  # First 3 authors
                    if 'LastName' in author and 'ForeName' in author:
                        authors.append(f"{author['LastName']}, {author['ForeName']}")
                author_string = "; ".join(authors)
                if len(author_list) > 3:
                    author_string += " et al."
                
                # Extract journal and year
                journal = medline_citation['Article']['Journal']['Title']
                pub_date = medline_citation['Article']['Journal']['JournalIssue']['PubDate']
                year = pub_date.get('Year', 'Unknown')
                
                # Extract PMID
                pmid = medline_citation['PMID']
                
                literature_data.append({
                    'pmid': pmid,
                    'title': title,
                    'authors': author_string,
                    'journal': journal,
                    'year': year,
                    'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                })
                
            except KeyError as e:
                print(f"Error parsing article: {e}")
                continue
        
        return pd.DataFrame(literature_data)
        
    except Exception as e:
        print(f"Error searching PubMed: {e}")
        return pd.DataFrame()

# Example usage
literature = search_pubmed_literature("CRISPR Cas9 bacteria", max_results=5)
print(literature[['title', 'authors', 'year']])
```

## Working with REST APIs
Interact with biological databases through REST APIs.

```python
import json
from urllib.parse import urlencode

class BiologicalAPI:
    """Base class for biological database APIs"""
    
    def __init__(self, base_url, rate_limit=1.0):
        self.base_url = base_url
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Python-Microbiology-Student/1.0'})
    
    def _make_request(self, endpoint, params=None):
        """Make API request with rate limiting"""
        
        url = f"{self.base_url}/{endpoint}"
        if params:
            url += f"?{urlencode(params)}"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Rate limiting
            time.sleep(self.rate_limit)
            
            return response.json()
            
        except requests.RequestException as e:
            print(f"API request failed: {e}")
            return None

class EnsemblAPI(BiologicalAPI):
    """Interface for Ensembl REST API"""
    
    def __init__(self):
        super().__init__("https://rest.ensembl.org", rate_limit=1.0)
    
    def get_gene_info(self, gene_id, species="homo_sapiens"):
        """Get gene information from Ensembl"""
        
        endpoint = f"lookup/id/{gene_id}"
        params = {'species': species, 'expand': 1}
        
        result = self._make_request(endpoint, params)
        
        if result:
            return {
                'gene_id': result.get('id'),
                'name': result.get('display_name'),
                'description': result.get('description'),
                'chromosome': result.get('seq_region_name'),
                'start': result.get('start'),
                'end': result.get('end'),
                'strand': result.get('strand')
            }
        return None
    
    def get_gene_sequence(self, gene_id, species="homo_sapiens"):
        """Get gene sequence from Ensembl"""
        
        endpoint = f"sequence/id/{gene_id}"
        params = {'type': 'genomic', 'species': species}
        
        result = self._make_request(endpoint, params)
        
        if result:
            return result.get('seq')
        return None

# Example usage
ensembl = EnsemblAPI()
gene_info = ensembl.get_gene_info("ENSG00000139618")  # BRCA2 gene
if gene_info:
    print(f"Gene: {gene_info['name']}")
    print(f"Description: {gene_info['description']}")
```

## Handling Rate Limits and Authentication
Implement proper rate limiting and authentication for API access.

```python
class RateLimitedSession:
    """Session with built-in rate limiting and retry logic"""
    
    def __init__(self, requests_per_second=1, max_retries=3):
        self.session = requests.Session()
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0
        self.max_retries = max_retries
    
    def get(self, url, **kwargs):
        """Rate-limited GET request with retry logic"""
        
        # Enforce rate limit
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            time.sleep(self.min_interval - time_since_last)
        
        # Attempt request with retries
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, **kwargs)
                self.last_request_time = time.time()
                
                # Handle rate limiting response
                if response.status_code == 429:  # Too Many Requests
                    retry_after = int(response.headers.get('Retry-After', 60))
                    print(f"Rate limited. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                return response
                
            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise
                print(f"Request failed (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return None

class AuthenticatedAPI:
    """API client with authentication support"""
    
    def __init__(self, base_url, api_key=None, username=None, password=None):
        self.base_url = base_url
        self.session = RateLimitedSession()
        
        # Set up authentication
        if api_key:
            self.session.session.headers.update({'Authorization': f'Bearer {api_key}'})
        elif username and password:
            self.session.session.auth = (username, password)
    
    def search_database(self, query, database="protein"):
        """Search authenticated database"""
        
        endpoint = f"{self.base_url}/search"
        params = {'q': query, 'database': database}
        
        try:
            response = self.session.get(endpoint, params=params)
            return response.json()
        except requests.RequestException as e:
            print(f"Database search failed: {e}")
            return None

# Example with environment variables for security
import os

api_key = os.getenv('BIOLOGICAL_DB_API_KEY')
if api_key:
    auth_api = AuthenticatedAPI("https://api.example-bio-db.org", api_key=api_key)
    results = auth_api.search_database("antibiotic resistance genes")
```

Web scraping and API integration enable automated data collection from biological databases, supporting large-scale research projects.
