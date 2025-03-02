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

# Example Useage
'''
data = pd.read_csv('your file path', header=0)
data.dropna(inplace=True)

# Run PhysioNexus directly with your parameters
G, causal_df = PhysioNexus(
    data=data,                                   # Your already loaded DataFrame
    exclude_cols=['Time[s]', 'Time[hh:mm:ss]'], # Exclude specific columns by name
    corr_threshold=0.6,                         # Custom correlation threshold
    f_stat_threshold=10,                        # Custom F-statistic threshold
    p_value_threshold=0.05,                     # Custom p-value threshold
    max_lag=3,                                  # Look at up to 2 lags
    output_dir='physionexus_results'
)

# Display the causal relationships (if any were found)
if causal_df is not None:
    print("Found causal relationships:")
    print(causal_df)
else:
    print("No causal relationships were found meeting the specified criteria.")

'''



