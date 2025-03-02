# 🧬 PhysioNexus Overview 
When athletes exercise intensely, their bodies respond with synchronized changes across multiple systems: heart rate and oxygen consumption increases, muscle oxygenation decreases, and muscle's electrical patterns oscillate. Understanding how these systems influence each other presents a significant challenge in exercise physiology. Traditional analysis methods examine these variables individually or use simple correlations, missing the critical question of causality – which physiological variables are driving changes in others, and which are merely responding?

PhysioNexus solves this problem by mapping cause-and-effect relationships in physiological time series data. Using Granger causality testing, it identifies predictive relationships between variables to transform complex measurements into clear network visualizations that show how systems interact.

Granger causality defines causation through prediction: if variable A "Granger-causes" variable B, then knowing the history of both A and B improves predictions of B's future values compared to using B's history alone. For example, if muscle oxygenation (SmO2) Granger-causes oxygen consumption (VO2), then previous SmO2 and VO2 values together better predict future VO2 than past VO2 measurements alone. Unlike correlation, which only shows that variables change together, Granger causality reveals directional relationships with time components.

The biological reality of feedback loops and circular relationships complicates this analysis, as many physiological systems influence each other simultaneously. PhysioNexus helps researchers navigate this complexity by quantifying and visualizing the strength and direction of these interconnected relationships.

# 🧬 How Does PhysioNexus Work?
## Overview 
PhysioNexus analyzes time series physiological data to reveal cause-and-effect relationships between variables like heart rate, oxygen consumption, and muscle oxygenation. The analysis follows a systematic approach:
- (1) Correlation Analysis: First, the program calculates correlations between all pairs of time series physiologic metrics.
- (2) Causality Testing: For pairs that exceed a specified correlation threshold, it performs Granger causality tests to determine potential causal relationships.
- (3) Network Construction: The results are used to build a directed graph where nodes represent physiological variables and edges represent causal relationships.
- (4) Visualization: The network is visualized with informative attributes that highlight relationship strength and direction, as explained in the next sub-section. 
- (5) Metric Calculation: Various network metrics are calculated to identify key influencers and relationship structures.

## Interpreting The Network Visualization
The network visualization transforms complex physiological relationships into an intuitive visual map. Arrows between variables show the direction of influence—which variable is causing changes in another. Blue connections indicate positive relationships (variables increase together), while red connections show negative relationships (one increases as another decreases). Thicker connections represent stronger correlations, and larger nodes indicate variables that influence many others, making key physiological drivers immediately apparent.

The statistical significance of each relationship is measured by its F-statistic and p-value. An F-statistic of 1-4 suggests weak causality with minimal predictive power, while 4-10 indicates moderate evidence with meaningful relationships. Strong causal evidence appears with F-statistics of 10-30, while values above 30 represent extremely strong causal relationships with overwhelming statistical significance (p << 0.001) and major predictive power.

## Network Metrics and Their Interpretation
PhysioNexus provides several metrics that reveal different aspects of the physiological system's structure:
- Out-degree reveals a variable's influence by counting how many other variables it affects. High out-degree variables act as system "drivers" or control points that regulate multiple physiological processes.
- In-degree shows how responsive a variable is by counting how many variables affect it. High in-degree variables represent integration points where multiple physiological signals converge.
- Degree centrality combines both measures to identify the most connected variables overall, highlighting the central parameters in the physiological network regardless of direction.
- Betweenness centrality identifies mediator variables that frequently appear on paths between other variables. These "brokers" represent critical intermediate steps in physiological cascades, connecting different systems or processes.

# 🧬 Implementation: A DIY Guide To Using PhysioNexus

PhysioNexus can be integrated into your analysis workflow in two ways:

## Method 1: Install as a Python Package (Recommended)
The most streamlined approach is to install PhysioNexus directly from GitHub using pip. This makes the function available for import in any of your Python environments without cluttering your code. Additionally, this method ensures you always have access to the latest version and keeps your analysis scripts clean and focused on your specific research questions.

You can install PhysioNexus as a Python package directly from Github using the code below:

```Bash
# Install the package directly from GitHub
!pip install git+https://github.com/evanpeikon/PhysioNexus.git 
```
```python
# Import and use
from PhysioNexus import PhysioNexus
```
```python
# Example Usaage
data = pd.read_csv('Path to your CSV file', header=0)

# Run PhysioNexus directly with custom parameters
G, causal_df = PhysioNexus(
    data=data,  
    exclude_cols=['Time[s]', 'Time[hh:mm:ss]'],   # Replace with your non-feature columns
    corr_threshold=0.6,                           # Correlation threshold for considering relationships
    f_stat_threshold=10,                          # F-statistic threshold for Granger causality
    p_value_threshold=0.05,                       # P-value threshold for statistical significance
    max_lag=3,                                    # Maximum lag to consider for Granger causality
    output_dir=None                               # Optional output directory to store results
)

# Display the causal relationships (if any were found)
if causal_df is not None:
    print("Found causal relationships:")
    print(causal_df)
else:
    print("No causal relationships were found meeting the specified criteria.")
```

## Method 2: Copy the Function Directly

For situations where you prefer to have all code self-contained or can't install packages, you can use PhysioNexus as a standard function by copying it directly into your project. This approach gives you the flexibility to modify the code for your specific needs but requires manually updating when improvements are made to the original function.

To implement this method simply copy the function copy below, then paste it into your Python script, Google Collab notebook, or Juypter notebook:

```python
# Import dependencies
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
import os

def PhysioNexus(data, exclude_cols=2, corr_threshold=0.6, f_stat_threshold=10, p_value_threshold=0.05, max_lag=2, output_dir=None):
    # Set output directory
    if output_dir is None:
        output_dir = '.'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Handle column exclusion
    if isinstance(exclude_cols, int):
        excluded_columns = data.columns[:exclude_cols]
        data_filtered = data.drop(columns=excluded_columns)
        print(f"Excluded first {exclude_cols} columns: {', '.join(excluded_columns)}")
    elif isinstance(exclude_cols, list):
        excluded_columns = exclude_cols
        data_filtered = data.drop(columns=excluded_columns, errors='ignore')
        print(f"Excluded columns: {', '.join(excluded_columns)}")
    else:
        raise ValueError("exclude_cols must be an integer or a list of column names")
    
    # Calculate correlation matrix using the filtered dataset ONLY
    correlation_matrix = data_filtered.corr(method='pearson')
    
    # Create a mask for correlations with absolute value >= threshold
    correlation_mask = np.abs(correlation_matrix) >= corr_threshold
    
    # Create a filtered correlation matrix for visualization
    filtered_corr = correlation_matrix.copy()
    filtered_corr[~correlation_mask] = 0
    
    # Create a directed network graph
    G = nx.DiGraph()
    
    # Add nodes ONLY from the filtered dataframe (after exclusions)
    for column in data_filtered.columns:
        G.add_node(column)
    
    # Function to check Granger causality between two time series
    def check_granger_causality(x, y, max_lag=max_lag):
        """
        Test if x Granger-causes y.
        Returns:
        - is_causal: Boolean indicating if x causes y
        - f_stat: F-statistic
        - p_value: p-value
        - lag: Optimal lag
        """
        try:
            # Test with specified max_lag
            test_result = grangercausalitytests(np.column_stack((y, x)), maxlag=max_lag, verbose=False)
            
            # Extract results for each lag
            f_stats = []
            p_values = []
            
            for lag in range(1, max_lag + 1):
                f_stat = test_result[lag][0]['ssr_ftest'][0]
                p_value = test_result[lag][0]['ssr_ftest'][1]
                f_stats.append(f_stat)
                p_values.append(p_value)
                
            # Find the highest F-statistic among all lags
            max_f_index = f_stats.index(max(f_stats))
            best_f = f_stats[max_f_index]
            best_p = p_values[max_f_index]
            best_lag = max_f_index + 1
            
            # Check if the best result meets our criteria
            is_causal = (best_f > f_stat_threshold) and (best_p < p_value_threshold)
            
            return is_causal, best_f, best_p, best_lag
        except:
            # Return False if there's an error (e.g., non-stationary data)
            return False, 0, 1, 0
    
    # Add edges based on correlation and Granger causality
    causal_edges = 0
    filtered_columns = data_filtered.columns # Make sure we only use columns from the filtered data
    
    print("Testing causal relationships...")
    for i, col1 in enumerate(filtered_columns):
        for j, col2 in enumerate(filtered_columns):
            # Don't test self-causality
            if i != j:
                corr_value = correlation_matrix.loc[col1, col2]
                # Only test causality if correlation meets threshold
                if abs(corr_value) >= corr_threshold:
                    # Check causality
                    a_causes_b, f_ab, p_ab, lag_ab = check_granger_causality(data_filtered[col1], data_filtered[col2])
                    
                    # Add edge if causality is detected
                    if a_causes_b:
                        G.add_edge(col1, col2, 
                                  weight=abs(corr_value),
                                  correlation=corr_value,
                                  color='red' if corr_value < 0 else 'blue',
                                  f_stat=f_ab,
                                  p_value=p_ab,
                                  lag=lag_ab)
                        causal_edges += 1
    
    print(f"\nCreated network with {G.number_of_nodes()} nodes and {causal_edges} causal edges")
    
    # After adding all edges, remove nodes without any connections
    nodes_to_remove = []
    for node in G.nodes():
        if G.degree(node) == 0:  # Node has no incoming or outgoing edges
            nodes_to_remove.append(node)
    
    G.remove_nodes_from(nodes_to_remove)
    if nodes_to_remove:
        print(f"\nRemoved {len(nodes_to_remove)} nodes without connections")
        if len(nodes_to_remove) <= 10:  # Only print names if there aren't too many
            print(f"Removed nodes: {', '.join(nodes_to_remove)}")
        else:
            print(f"Removed nodes: {', '.join(nodes_to_remove[:5])}... and {len(nodes_to_remove)-5} more")
    
    # 5. Visualize the correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    
    # Visualize the network
    plt.figure(figsize=(16, 14))
    
    # Only visualize if there are nodes in the network
    if G.number_of_nodes() > 0:
        # Use spring layout for better visualization
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        
        # Get edge colors, weights, and F-statistics
        edge_colors = [G[u][v]['color'] for u, v in G.edges()]
        edge_weights = [abs(G[u][v]['correlation']) * 2 for u, v in G.edges()]  # Multiply by 2 to make edges more visible
        
        # Adjust node size based on how many other variables it causes (out-degree)
        node_size = [300 * (1 + G.out_degree(node)) for node in G.nodes()]
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='lightblue', alpha=0.8)
        
        # Draw directed edges with appropriate colors and arrowheads
        nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color=edge_colors, alpha=0.7, arrows=True, arrowstyle='->', arrowsize=10, connectionstyle='arc3,rad=0.1')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        
        plt.title(f'Causal Network (Correlation ≥ {corr_threshold}, F-statistic > {f_stat_threshold}, p < {p_value_threshold})')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'causal_network.png'))
        plt.show()
    else:
        plt.close()
        print("No causal relationships found meeting the criteria. Network visualization skipped.")
    
    # Calculate network metrics
    print("\nNetwork Summary:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    
    # Calculate and print node centrality measures
    if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
        # Out-degree (causal influence)
        print("\nTop 5 nodes by out-degree (causal influence):")
        out_degree_dict = dict(G.out_degree())
        sorted_out_degree = sorted(out_degree_dict.items(), key=lambda x: x[1], reverse=True)
        for node, degree in sorted_out_degree[:min(5, len(sorted_out_degree))]:
            if degree > 0:  # Only show nodes with outgoing connections
                print(f"{node}: {degree} outgoing connections")
        
        # In-degree (influenced by others)
        print("\nTop 5 nodes by in-degree (influenced by others):")
        in_degree_dict = dict(G.in_degree())
        sorted_in_degree = sorted(in_degree_dict.items(), key=lambda x: x[1], reverse=True)
        for node, degree in sorted_in_degree[:min(5, len(sorted_in_degree))]:
            if degree > 0:  # Only show nodes with incoming connections
                print(f"{node}: {degree} incoming connections")
        
        # Degree centrality (overall connection importance)
        print("\nTop 5 nodes by degree centrality (overall connection importance):")
        degree_centrality = nx.degree_centrality(G)
        sorted_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
        for node, centrality in sorted_degree[:min(5, len(sorted_degree))]:
            print(f"{node}: {centrality:.4f}")
        
        # Betweenness centrality (information flow brokers)
        print("\nTop 5 nodes by betweenness centrality (information flow brokers):")
        betweenness_centrality = nx.betweenness_centrality(G)
        sorted_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
        for node, centrality in sorted_betweenness[:min(5, len(sorted_betweenness))]:
            print(f"{node}: {centrality:.4f}")
            
        # Strongest causal relationships
        print("\nTop 5 strongest causal relationships by F-statistic:")
        sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['f_stat'], reverse=True)
        for u, v, data in sorted_edges[:min(5, len(sorted_edges))]:
            print(f"{u} → {v}: F={data['f_stat']:.2f}, p={data['p_value']:.5f}, correlation={data['correlation']:.3f}")
    
    # Generate a table of all causal relationships
    causal_df = None
    if G.number_of_edges() > 0:
        causal_relationships = []
        for u, v, data in G.edges(data=True):
            causal_relationships.append((
                u,  # Cause
                v,  # Effect
                data['correlation'],  # Correlation value
                "Positive" if data['correlation'] > 0 else "Negative",  # Correlation type
                data['f_stat'],  # F-statistic
                data['p_value'],  # p-value
                data['lag']))  # Lag with strongest effect)
        
        # Convert to DataFrame and sort by F-statistic
        causal_df = pd.DataFrame(causal_relationships, columns=["Cause", "Effect", "Correlation", "Correlation Type",  "F-statistic", "p-value", "Optimal Lag"])
        causal_df = causal_df.sort_values(by="F-statistic", ascending=False)
        
    return G, causal_df
```
```python
# Example Usage
data = pd.read_csv('YOUR FILE PATH', header=0)

# Execute PhysioNexus
G, causal_df = PhysioNexus(
    data=data,                             # Your already loaded DataFrame
    exclude_cols= ['Column1', 'Column2'], # Exclude specific columns by name
    corr_threshold=0.6,                   # Custom correlation threshold
    f_stat_threshold=15,                  # Custom F-statistic threshold
    p_value_threshold=0.05,               # Custom p-value threshold
    max_lag=2)                            # Look at up to 2 lags
```

# 🧬 Elite Athlete Case Study
To demonstrate the tool's capabilities, we'll apply it to a case study using data from an athlete performing a ramp incremental exercise test to exhaustion. In this test, the athlete 
exercised at a progressively increasing intensity level until they could no longer continue, allowing us to observe the full spectrum of physiological responses from rest to maximal exertion. The data from this test was loaded into a pandas DataFrame and contains time series measurements for several physiological metrics:

- VO2 (oxygen consumption)
- Heart rate 
- Power output
- Muscle recruitment 
- Respiratory rate 
- Tidal volume 
- Ventilatory exchange 
- Cerebral oxygenation
- Heart rate variability
- Muscle oxygenation
- Blood lactate

> Note: This case study assumes you've installed PhysioNexus using Method 1, from the implementation section above, though the functionality is identical with either implementation approach.

```python
# Install the PhysioNexus package directly from GitHub
!pip install git+https://github.com/evanpeikon/PhysioNexus.git

# Import PhysioNexus
from PhysioNexus import PhysioNexus

# Load Data
data = pd.read_csv('ramp_incremental_test.csv', header=0)
data.dropna(inplace=True) # Remove rows / columns with missing values 

# Run PhysioNexus directly with custom parameters
G, causal_df = PhysioNexus(
    data=data,  
    exclude_cols=['Time[s]', 'Time[hh:mm:ss]'],
    corr_threshold=0.6,
    f_stat_threshold=10,
    p_value_threshold=0.05,
    max_lag=3,
    output_dir=None
)

'''
# Optional: Display the causal relationships (if any were found)
if causal_df is not None:
    print("Found causal relationships:")
    print(causal_df)
else:
    print("No causal relationships were found meeting the specified criteria.")
'''
```
The code above produced the following network and metrics:

<img width="939" alt="Screenshot 2025-02-26 at 2 51 13 PM" src="https://github.com/user-attachments/assets/7e6de768-65b2-4a28-9672-5b1861061150" />

```
Network Summary:
Number of nodes: 9
Number of edges: 43

Top 5 nodes by out-degree (causal influence):
SmO2: 7 outgoing connections
Blood_Lactate: 7 outgoing connections
Ventilatory_Exchange: 6 outgoing connections
HR[bpm]: 5 outgoing connections
Cerebral_O2: 5 outgoing connections

Top 5 nodes by in-degree (influenced by others):
HR[bpm]: 6 incoming connections
Tidal_Volume: 6 incoming connections
Respiration_Rate: 5 incoming connections
RR[ms]: 5 incoming connections
SmO2: 5 incoming connections

Top 5 nodes by degree centrality (overall connection importance):
SmO2: 1.5000
Blood_Lactate: 1.5000
HR[bpm]: 1.3750
Ventilatory_Exchange: 1.2500
Respiration_Rate: 1.1250

Top 5 nodes by betweenness centrality (information flow brokers):
SmO2: 0.1384
Blood_Lactate: 0.1384
Respiration_Rate: 0.0711
HR[bpm]: 0.0610
Tidal_Volume: 0.0295

Top 5 strongest causal relationships by F-statistic:
HR[bpm] → RR[ms]: F=107.39, p=0.00000, correlation=-0.922
HR[bpm] → Ventilatory_Exchange: F=100.49, p=0.00000, correlation=0.926
RR[ms] → Tidal_Volume: F=72.18, p=0.00000, correlation=-0.767
HR[bpm] → Tidal_Volume: F=69.79, p=0.00000, correlation=0.788
Ventilatory_Exchange → Respiration_Rate: F=65.86, p=0.00000, correlation=0.846
```

## Key Insights from PhysioNexus Network Analysis

### Central Regulators and System Integration
The network analysis reveals a physiological system with clear organization and multiple levels of regulation during incremental exercise. Muscle oxygenation (SmO2) and blood lactate emerge as the most influential variables, each influencing seven other physiological parameters. This positions them as master regulators of the exercise response, suggesting that local muscle conditions drive systemic adaptations rather than simply responding to them.

Heart rate (HR) shows a fascinating dual role as both a major driver (5 outgoing connections) and a highly responsive variable (6 incoming connections). This reflects its position as both a control parameter and an integration point for multiple physiological inputs - a biological control hub that both influences and is influenced by the overall physiological state.

### Respiratory Chain of Command
The analysis exposes a clear respiratory control hierarchy, with ventilatory exchange driving respiration rate (F=65.86, extremely strong causality). This confirms established understanding of ventilatory control during exercise, where total minute ventilation increases drive breathing frequency rather than the reverse.

The strong causal relationship from heart rate to ventilatory exchange (F=100.49) demonstrates the tight cardiorespiratory coupling necessary for effective oxygen delivery and carbon dioxide removal during exercise. This cardiovascular-to-respiratory causality suggests that cardiac output drives ventilatory responses rather than the reverse during incremental exercise.

### Autonomic Nervous System Dynamics
The strongest causal relationship in the entire network is from heart rate to heart rate variability (RR[ms]) with an F-statistic of 107.39 and a strong negative correlation of -0.922. This powerful inverse relationship reflects the progressive withdrawal of parasympathetic tone and increase in sympathetic drive as exercise intensity increases, a fundamental aspect of autonomic control during exercise.

### Tissue Oxygenation as An Information Broker
Both muscle oxygenation (SmO2) and blood lactate show the highest betweenness centrality (0.1384), indicating they serve as critical information brokers in the physiological network. This suggests these variables connect different physiological subsystems - linking local muscle metabolism with systemic cardiovascular and respiratory responses.

The high centrality of these tissue oxygenation markers aligns with current understanding that peripheral chemoreceptors and metabolic sensors in active muscle tissue provide critical feedback that drives cardiorespiratory adjustments during exercise.

### Clinical and Training Implications
These network results suggest that interventions targeting muscle oxygenation and blood lactate might have the most widespread effects across multiple physiological systems. For athletes and coaches, this suggests that training strategies focused on improving muscle oxygen utilization or lactate handling may have cascading benefits across cardiorespiratory function.
The dual role of heart rate as both causal driver and recipient highlights why heart rate monitoring remains such an effective and comprehensive training metric - it both influences and reflects overall physiological status during exercise.

The analysis also shows that respiration rate, while an important variable, is primarily a downstream responder rather than an initiating driver. This suggests that breathing frequency training interventions might be less effective than those targeting upstream variables like ventilatory exchange or heart rate.
Overall, this network analysis provides evidence for a hierarchical yet interconnected physiological control system during exercise, where local muscle conditions drive systemic cardiovascular and respiratory responses through complex feedback and feed-forward mechanisms.

# 🧬 Potential Applications, Limitations, and Future Work 
## Potential Applications 
This causal network analysis tool can be applied in various contexts. For training optimization, identifying key physiological drivers can help focus interventions on the most influential variables rather than those that merely respond to other changes. Individual profiling becomes possible by comparing causal networks between athletes, potentially revealing individual differences in physiological regulation that could inform personalized training approaches. The tool also enables intervention assessment by analyzing causal networks before and after training programs or nutritional strategies to reveal how interventions alter physiological regulation mechanisms.

Changes in the causal structure might serve as early indicators of fatigue or overtraining, potentially providing a novel way to monitor athlete status. By comparing causal networks across different exercise modalities, researchers can reveal sport-specific physiological regulation patterns that inform specialized training strategies. Beyond sports, this approach could be used to analyze physiological dysregulation in clinical populations, offering insights into disease mechanisms or treatment responses.

## Limitations and Future Work
While powerful, this approach has several limitations to consider. Granger causality assumes linearity and stationarity, which may not fully capture complex physiological relationships that often involve nonlinear dynamics and time-varying parameters. The analysis is sensitive to the sampling rate of the data, and different causal relationships may emerge at different time scales, potentially missing important connections that operate on very fast or slow timescales. Additionally, unmeasured variables could drive apparent causal relationships between measured variables, creating misleading connections in the network.

Future work could address these limitations through several avenues. Implementing information-theoretic or nonlinear causality measures would better capture the complex nonlinear interactions in physiological systems. Analyzing causality at different temporal resolutions would help identify relationships that operate at various timescales. Adding physiological constraints to the causal discovery process would incorporate domain knowledge to guide the analysis. Examining how causal networks evolve during different exercise phases would capture the dynamic nature of physiological regulation during changing conditions.

# 🧬 Contributing and Support
PhysioNexus is an open-source project and welcomes contributions from the community. If you encounter issues, have suggestions for improvements, or would like to contribute to the project, feel free to reach out: evanpeikon@gmail.com.
