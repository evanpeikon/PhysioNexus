import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.api as sm
import os
import scipy.stats
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go

def transfer_entropy(x, y, k=5, max_lag=2):
    """
    Calculate transfer entropy which can detect nonlinear causal relationships
    
    Parameters:
    - x: Potential cause time series
    - y: Effect time series
    - k: Number of nearest neighbors
    - max_lag: Maximum lag to test
    
    Returns:
    - te: Transfer entropy
    - normalized_te: Normalized transfer entropy
    - p_value: p-value from surrogate testing
    - best_lag: Lag with highest transfer entropy
    """
    best_te = 0
    best_norm_te = 0
    best_p = 1
    best_lag = 0
    
    for lag in range(1, max_lag + 1):
        # Create lagged versions
        x_lagged = x[:-lag].values.reshape(-1, 1)
        y_lagged = y[:-lag].values.reshape(-1, 1)
        y_current = y[lag:].values.reshape(-1, 1)
        
        # Calculate entropies using KNN approach
        k = min(k, len(x_lagged) - 1)  # Ensure k is not too large
        
        # Joint spaces
        xy = np.hstack([x_lagged, y_lagged])
        xyz = np.hstack([x_lagged, y_lagged, y_current])
        
        # Find k-nearest neighbors
        knn_xy = NearestNeighbors(n_neighbors=k+1).fit(xy)
        dist_xy, _ = knn_xy.kneighbors(xy)
        epsilon_xy = dist_xy[:, -1]  # Distance to kth neighbor
        
        knn_xyz = NearestNeighbors(n_neighbors=k+1).fit(xyz)
        dist_xyz, _ = knn_xyz.kneighbors(xyz)
        epsilon_xyz = dist_xyz[:, -1]
        
        # Calculate entropy terms
        n = len(x_lagged)
        te = np.mean(np.log(epsilon_xy/epsilon_xyz)) + np.log(n)
        
        # Normalize by the entropy of Y
        knn_y = NearestNeighbors(n_neighbors=k+1).fit(y_current)
        dist_y, _ = knn_y.kneighbors(y_current)
        epsilon_y = dist_y[:, -1]
        h_y = np.mean(np.log(epsilon_y)) + np.log(n)
        norm_te = te / h_y if h_y != 0 else 0
        
        # Calculate p-value through permutation testing
        num_surrogates = 100
        surrogate_tes = []
        
        for _ in range(num_surrogates):
            # Shuffle x to destroy causality
            x_shuffle = np.random.permutation(x_lagged)
            
            # Calculate TE with shuffled data
            xy_shuf = np.hstack([x_shuffle, y_lagged])
            xyz_shuf = np.hstack([x_shuffle, y_lagged, y_current])
            
            knn_xy_shuf = NearestNeighbors(n_neighbors=k+1).fit(xy_shuf)
            dist_xy_shuf, _ = knn_xy_shuf.kneighbors(xy_shuf)
            epsilon_xy_shuf = dist_xy_shuf[:, -1]
            
            knn_xyz_shuf = NearestNeighbors(n_neighbors=k+1).fit(xyz_shuf)
            dist_xyz_shuf, _ = knn_xyz_shuf.kneighbors(xyz_shuf)
            epsilon_xyz_shuf = dist_xyz_shuf[:, -1]
            
            te_shuf = np.mean(np.log(epsilon_xy_shuf/epsilon_xyz_shuf)) + np.log(n)
            surrogate_tes.append(te_shuf)
        
        p_value = np.mean(np.array(surrogate_tes) >= te)
        
        if te > best_te:
            best_te = te
            best_norm_te = norm_te
            best_p = p_value
            best_lag = lag
    
    return best_te, best_norm_te, best_p, best_lag

def check_multivariate_granger_causality(data, target, predictors, max_lag=2, f_stat_threshold=10, p_value_threshold=0.05):
    """
    Test if multiple variables jointly Granger-cause a target variable.
    
    Parameters:
    - data: DataFrame containing all variables
    - target: Target variable name
    - predictors: List of predictor variable names
    - max_lag: Maximum lag to test
    
    Returns:
    - is_causal: Boolean indicating if predictors cause target
    - f_stat: F-statistic
    - p_value: p-value
    - lag: Optimal lag
    """
    best_f = 0
    best_p = 1
    best_lag = 0
    is_causal = False
    
    try:
        # Test each lag from 1 to max_lag
        for lag in range(1, max_lag + 1):
            # Create lagged predictors
            X = pd.DataFrame()
            y = data[target][lag:]
            
            # Add lagged values for each predictor
            for pred in predictors:
                for l in range(1, lag + 1):
                    X[f"{pred}_lag{l}"] = data[pred].shift(l)[lag:]
            
            # Also include lagged values of the target variable
            for l in range(1, lag + 1):
                X[f"{target}_lag{l}"] = data[target].shift(l)[lag:]
            
            # Remove any NaN rows
            X = X.dropna()
            y = y[:len(X)]
            
            # Create two models: one with and one without the predictor variables
            restricted_cols = [col for col in X.columns if col.startswith(f"{target}_lag")]
            if len(restricted_cols) > 0:  # Only proceed if there are target lags
                model_restricted = sm.OLS(y, X[restricted_cols]).fit()
                model_unrestricted = sm.OLS(y, X).fit()
                
                # F-test for nested models
                df1 = model_restricted.df_resid - model_unrestricted.df_resid
                df2 = model_unrestricted.df_resid
                
                if df1 > 0 and df2 > 0:  # Only calculate if valid degrees of freedom
                    f_stat = ((model_restricted.ssr - model_unrestricted.ssr) / df1) / \
                            (model_unrestricted.ssr / df2)
                    
                    p_value = 1 - scipy.stats.f.cdf(f_stat, df1, df2)
                    
                    if f_stat > best_f:
                        best_f = f_stat
                        best_p = p_value
                        best_lag = lag
        
        is_causal = (best_f > f_stat_threshold) and (best_p < p_value_threshold)
        return is_causal, best_f, best_p, best_lag
    except Exception as e:
        print(f"Error in multivariate testing: {e}")
        return False, 0, 1, 0

def check_granger_causality(x, y, max_lag=2, f_stat_threshold=10, p_value_threshold=0.05):
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

def visualize_enhanced_network(G, output_dir):
    """
    Enhanced network visualization with differentiation between relationship types
    """
    plt.figure(figsize=(18, 16))
    
    # Define node colors based on type
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        if G.nodes[node].get('node_type') == 'group':
            node_colors.append('lightgreen')  # Group nodes
            node_sizes.append(500)  # Larger for groups
        else:
            node_colors.append('lightblue')  # Regular variables
            node_sizes.append(300 * (1 + G.out_degree(node)))
    
    # Separate edges by type
    linear_edges = [(u, v) for u, v in G.edges() if 'te' not in G[u][v] and G.nodes[u].get('node_type') != 'group']
    nonlinear_edges = [(u, v) for u, v in G.edges() if 'te' in G[u][v]]
    multivar_edges = [(u, v) for u, v in G.edges() if G.nodes[u].get('node_type') == 'group']
    
    # Edge colors and styles
    linear_colors = [G[u][v]['color'] for u, v in linear_edges]
    nonlinear_colors = ['green' for _ in nonlinear_edges]
    multivar_colors = ['purple' for _ in multivar_edges]
    
    # Create layout
    pos = nx.spring_layout(G, k=0.6, iterations=100, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
    
    # Draw different types of edges
    if linear_edges:
        nx.draw_networkx_edges(G, pos, edgelist=linear_edges, width=2, 
                              edge_color=linear_colors, style='solid', 
                              alpha=0.7, arrows=True, arrowstyle='->', arrowsize=10,
                              connectionstyle='arc3,rad=0.1')
    
    if nonlinear_edges:
        nx.draw_networkx_edges(G, pos, edgelist=nonlinear_edges, width=2,
                              edge_color=nonlinear_colors, style='dashed',
                              alpha=0.7, arrows=True, arrowstyle='->', arrowsize=10,
                              connectionstyle='arc3,rad=0.1')
    
    if multivar_edges:
        nx.draw_networkx_edges(G, pos, edgelist=multivar_edges, width=3,
                              edge_color=multivar_colors, style='dotted',
                              alpha=0.7, arrows=True, arrowstyle='->', arrowsize=15,
                              connectionstyle='arc3,rad=0.1')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    # Create a legend
    from matplotlib.lines import Line2D
    legend_elements = []
    
    if linear_edges:
        legend_elements.extend([
            Line2D([0], [0], color='blue', lw=2, label='Positive Linear Causality'),
            Line2D([0], [0], color='red', lw=2, label='Negative Linear Causality')
        ])
    
    if nonlinear_edges:
        legend_elements.append(
            Line2D([0], [0], color='green', linestyle='dashed', lw=2, label='Nonlinear Causality')
        )
    
    if multivar_edges:
        legend_elements.append(
            Line2D([0], [0], color='purple', linestyle='dotted', lw=3, label='Multivariate Causality')
        )
    
    if legend_elements:
        plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title('Enhanced Causal Network')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'enhanced_causal_network.png'))
    plt.close()

def create_alluvial_diagram(G, output_dir):
    """
    Create an alluvial (Sankey) diagram showing causal flows
    """
    if G.number_of_nodes() == 0:
        print("No nodes in graph. Skipping alluvial diagram.")
        return
        
    # Create source, target, and value lists for Sankey diagram
    source = []
    target = []
    value = []
    label = []
    
    # Get unique nodes and assign indices
    all_nodes = list(G.nodes())
    node_dict = {node: i for i, node in enumerate(all_nodes)}
    
    # Create node labels
    label = all_nodes
    
    # Create links
    for u, v, data in G.edges(data=True):
        source.append(node_dict[u])
        target.append(node_dict[v])
        
        # Use appropriate value based on edge type
        if 'f_stat' in data:
            value.append(data['f_stat'])
        elif 'te' in data:
            value.append(data['te'] * 10)  # Scale transfer entropy
        else:
            value.append(abs(data.get('correlation', 0.5)) * 10)  # Default fallback
    
    # Create figure
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=label
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        )
    )])
    
    fig.update_layout(title_text="Causal Flow Diagram", font_size=10)
    fig.write_html(os.path.join(output_dir, 'causal_flow.html'))
    print(f"Alluvial diagram saved to {os.path.join(output_dir, 'causal_flow.html')}")

def create_causal_matrix(G, output_dir):
    """
    Create a heatmap visualization of the causal strength matrix
    """
    if G.number_of_nodes() == 0:
        print("No nodes in graph. Skipping causal matrix.")
        return
        
    # Get all nodes
    nodes = list(G.nodes())
    n = len(nodes)
    
    # Create causal adjacency matrix
    causal_matrix = np.zeros((n, n))
    
    # Fill matrix with causality strengths
    for i, source in enumerate(nodes):
        for j, target in enumerate(nodes):
            if G.has_edge(source, target):
                if 'f_stat' in G[source][target]:
                    causal_matrix[i, j] = G[source][target]['f_stat']
                elif 'te' in G[source][target]:
                    causal_matrix[i, j] = G[source][target]['te'] * 10  # Scale for visibility
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(causal_matrix, annot=True, fmt=".1f", cmap="YlGnBu",
               xticklabels=nodes, yticklabels=nodes)
    plt.title('Causal Strength Matrix')
    plt.xlabel('Effect')
    plt.ylabel('Cause')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'causal_matrix.png'))
    plt.close()
    print(f"Causal matrix saved to {os.path.join(output_dir, 'causal_matrix.png')}")

def PhysioNexus(data, exclude_cols=2, corr_threshold=0.6, f_stat_threshold=10, 
                p_value_threshold=0.05, max_lag=2, output_dir=None, 
                causality_type='linear', k_neighbors=5, multivariate_groups=None):
    """
    Enhanced PhysioNexus function with multiple causality detection options
    
    Parameters:
    - data: DataFrame with time series data
    - exclude_cols: Integer (first n columns) or list of column names to exclude
    - corr_threshold: Minimum absolute correlation to consider for causality testing
    - f_stat_threshold: Minimum F-statistic for significance
    - p_value_threshold: Maximum p-value for significance
    - max_lag: Maximum lag to test for causality
    - output_dir: Directory to save visualizations
    - causality_type: One of 'linear', 'nonlinear', or 'both'
    - k_neighbors: Parameter for transfer entropy calculation
    - multivariate_groups: Dictionary mapping target variables to lists of potential causal variables
    
    Returns:
    - G: NetworkX DiGraph of causal relationships
    - causal_df: DataFrame with all causal relationships found
    """
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
    
    # Add edges based on correlation and causality
    causal_edges = 0
    filtered_columns = data_filtered.columns  # Make sure we only use columns from the filtered data
    
    # Test for linear causality (Granger)
    if causality_type in ['linear', 'both']:
        print("Testing linear causal relationships using Granger causality...")
        for i, col1 in enumerate(filtered_columns):
            for j, col2 in enumerate(filtered_columns):
                # Don't test self-causality
                if i != j:
                    corr_value = correlation_matrix.loc[col1, col2]
                    # Only test causality if correlation meets threshold
                    if abs(corr_value) >= corr_threshold:
                        # Check causality
                        a_causes_b, f_ab, p_ab, lag_ab = check_granger_causality(
                            data_filtered[col1], data_filtered[col2], 
                            max_lag=max_lag, 
                            f_stat_threshold=f_stat_threshold, 
                            p_value_threshold=p_value_threshold
                        )
                        
                        # Add edge if causality is detected
                        if a_causes_b:
                            G.add_edge(col1, col2, 
                                      weight=abs(corr_value),
                                      correlation=corr_value,
                                      color='red' if corr_value < 0 else 'blue',
                                      f_stat=f_ab,
                                      p_value=p_ab,
                                      lag=lag_ab,
                                      causality_type='linear')
                            causal_edges += 1
    
    # Test for nonlinear causality (Transfer Entropy)
    if causality_type in ['nonlinear', 'both']:
        print("Testing nonlinear causal relationships using transfer entropy...")
        for i, col1 in enumerate(filtered_columns):
            for j, col2 in enumerate(filtered_columns):
                if i != j:
                    corr_value = correlation_matrix.loc[col1, col2]
                    if abs(corr_value) >= corr_threshold:
                        # Skip if we already found linear causality and don't want to duplicate
                        if causality_type == 'both' and G.has_edge(col1, col2):
                            continue
                            
                        te, norm_te, p_value, lag = transfer_entropy(
                            data_filtered[col1], data_filtered[col2], 
                            k=k_neighbors, max_lag=max_lag
                        )
                        
                        # Use transfer entropy threshold and p-value for significance
                        if norm_te > 0.05 and p_value < p_value_threshold:
                            G.add_edge(col1, col2,
                                      weight=abs(corr_value),
                                      correlation=corr_value,
                                      color='green',  # Use green for nonlinear
                                      te=te,
                                      norm_te=norm_te,
                                      p_value=p_value,
                                      lag=lag,
                                      causality_type='nonlinear')
                            causal_edges += 1
    
    # Add multivariate causality if specified
    if multivariate_groups:
        print("Testing multivariate causal relationships...")
        for target, predictors in multivariate_groups.items():
            # Skip if target or any predictor isn't in the filtered data
            if target not in data_filtered.columns:
                continue
                
            valid_predictors = [p for p in predictors if p in data_filtered.columns]
            if not valid_predictors:
                continue
                
            is_causal, f_stat, p_value, lag = check_multivariate_granger_causality(
                data_filtered, target, valid_predictors, max_lag=max_lag,
                f_stat_threshold=f_stat_threshold, p_value_threshold=p_value_threshold
            )
            
            if is_causal:
                # Create a "meta node" representing the group
                group_name = f"Group({','.join(valid_predictors[:2])}{'...' if len(valid_predictors) > 2 else ''})"
                G.add_node(group_name, node_type='group', members=valid_predictors)
                
                # Add edge from group to target
                G.add_edge(group_name, target,
                          weight=1.0,  # Default weight for multivariate
                          correlation=None,  # No single correlation value
                          color='purple',  # Different color for multivariate
                          f_stat=f_stat,
                          p_value=p_value,
                          lag=lag,
                          causality_type='multivariate')
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
    
    # Create enhanced visualizations
    if G.number_of_nodes() > 0:
        # Enhanced network visualization
        visualize_enhanced_network(G, output_dir)
        
        # Alluvial diagram
        create_alluvial_diagram(G, output_dir)
        
        # Causal matrix
        create_causal_matrix(G, output_dir)
    else:
        print("No causal relationships found meeting the criteria. Visualizations skipped.")
    
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
        linear_edges = [(u, v, data) for u, v, data in G.edges(data=True) 
                        if data.get('causality_type') == 'linear']
        nonlinear_edges = [(u, v, data) for u, v, data in G.edges(data=True) 
                          if data.get('causality_type') == 'nonlinear']
        multivar_edges = [(u, v, data) for u, v, data in G.edges(data=True) 
                         if data.get('causality_type') == 'multivariate']
        
        if linear_edges:
            print("\nTop linear causal relationships by F-statistic:")
            sorted_linear = sorted(linear_edges, key=lambda x: x[2]['f_stat'], reverse=True)
            for u, v, data in sorted_linear[:min(5, len(sorted_linear))]:
                print(f"{u} → {v}: F={data['f_stat']:.2f}, p={data['p_value']:.5f}, correlation={data.get('correlation', 'N/A')}")
        
        if nonlinear_edges:
            print("\nTop nonlinear causal relationships by transfer entropy:")
            sorted_nonlinear = sorted(nonlinear_edges, key=lambda x: x[2]['te'], reverse=True)
            for u, v, data in sorted_nonlinear[:min(5, len(sorted_nonlinear))]:
                print(f"{u} → {v}: TE={data['te']:.4f}, norm_TE={data['norm_te']:.4f}, p={data['p_value']:.5f}")
        
        if multivar_edges:
            print("\nMultivariate causal relationships:")
            for u, v, data in multivar_edges[:min(5, len(multivar_edges))]:
                print(f"{u} → {v}: F={data['f_stat']:.2f}, p={data['p_value']:.5f}, lag={data['lag']}")
    
    # Generate a table of all causal relationships
    causal_df = None
    if G.number_of_edges() > 0:
        causal_relationships = []
        for u, v, data in G.edges(data=True):
            relationship_type = data.get('causality_type', 'unknown')
            
            # Build row based on relationship type
            if relationship_type == 'linear':
                causal_relationships.append((
                    u,  # Cause
                    v,  # Effect
                    data.get('correlation', None),  # Correlation value
                    "Positive" if data.get('correlation', 0) > 0 else "Negative",  # Correlation type
                    data.get('f_stat', None),  # F-statistic
                    data.get('p_value', None),  # p-value
                    data.get('lag', None),     # Optimal lag
                    "Linear"))                 # Causality type
            elif relationship_type == 'nonlinear':
                causal_relationships.append((
                    u,  # Cause
                    v,  # Effect
                    data.get('correlation', None),  # Correlation value
                    "Positive" if data.get('correlation', 0) > 0 else "Negative",  # Correlation type
                    data.get('te', None),      # Transfer entropy
                    data.get('p_value', None), # p-value
                    data.get('lag', None),     # Optimal lag
                    "Nonlinear"))              # Causality type
            elif relationship_type == 'multivariate':
                causal_relationships.append((
                    u,  # Cause (group)
                    v,  # Effect
                    None,                      # No single correlation
                    "Group",                   # Correlation type
                    data.get('f_stat', None),  # F-statistic
                    data.get('p_value', None), # p-value
                    data.get('lag', None),     # Optimal lag
                    "Multivariate"))           # Causality type
        
        # Convert to DataFrame and sort by causality strength
        causal_df = pd.DataFrame(causal_relationships, 
                                 columns=["Cause", "Effect", "Correlation/TE", "Correlation Type", 
                                         "Strength", "p-value", "Optimal Lag", "Causality Type"])
        
        # Sort by strength (whether F-stat or TE)
        causal_df = causal_df.sort_values(by="Strength", ascending=False)
        
    return G, causal_df



