# 🧬 PhysioNexus: Mapping Cause and Effect In Time-Series Physiological Data
During intense exercise our bodies responds with synchronized changes across multiple systems: heart rate and oxygen consumption increase, muscle oxygenation decreases, and muscle's electrical patterns oscillate. Understanding how these systems influence each other presents a significant challenge in exercise physiology. Traditional analysis methods examine these variables individually or use simple correlations, missing the critical question of causality – which physiological variables are driving changes in others, and which are merely responding?

PhysioNexus solves this problem by mapping cause-and-effect relationships in physiological time series data. Using both Granger causality and multivariate testing approaches, it identifies predictive relationships between variables to transform complex measurements into clear network visualizations that show how systems interact.

Granger causality defines causation through prediction: if variable A "Granger-causes" variable B, then knowing the history of both A and B improves predictions of B's future values compared to using B's history alone. For example, if muscle oxygenation (SmO2) Granger-causes oxygen consumption (VO2), then previous SmO2 and VO2 values together better predict future VO2 than past VO2 measurements alone. Unlike correlation, which only shows that variables change together, Granger causality reveals directional relationships with time components.

The biological reality of feedback loops and circular relationships complicates this analysis, as many physiological systems influence each other simultaneously. PhysioNexus helps researchers navigate this complexity by quantifying and visualizing the strength and direction of these interconnected relationships through multiple visualization approaches, including interactive network graphs, causal matrix heatmaps, and causal flow diagrams. Additionally, the multivariate causality testing capability allows researchers to detect when combinations of variables jointly influence a target variable - an essential feature for understanding complex physiological systems where multiple signals might work together to drive changes in other systems.

# 🧬 How Does PhysioNexus Work?
## Overview 
PhysioNexus analyzes time series physiological data to reveal cause-and-effect relationships between variables like heart rate, oxygen consumption, and muscle oxygenation. The analysis follows a systematic approach:

<img width="1072" alt="Screenshot 2025-04-17 at 1 11 01 PM" src="https://github.com/user-attachments/assets/c2406218-d9f0-4f89-8928-30f2835a2bb2" />

- (1) Correlation Analysis: First, the program calculates correlations between all pairs of time series physiologic metrics.
- (2) Causality Testing: For pairs that exceed a specified correlation threshold, it performs linear Granger causality tests and/or multivariate causality tests to determine potential causal relationships.
- (3) Network Construction: The results are used to build a directed graph where nodes represent physiological variables and edges represent causal relationships. For multivariate relationships, special group nodes represent combinations of variables that jointly cause changes in a target variable.
- (4) Visualization: The network is visualized with informative attributes that highlight relationship strength and direction, through multiple visualization types including network graphs, causal matrices, and flow diagrams.
- (5) Metric Calculation: Various network metrics are calculated to identify key influencers and relationship structures.

## Interpreting The Network Visualization

<img width="359" alt="Screenshot 2025-04-17 at 1 12 10 PM" src="https://github.com/user-attachments/assets/73225336-2504-489e-9479-ab7db432b62d" />

The network visualization transforms complex physiological relationships into an intuitive visual map. Arrows between variables show the direction of influence—which variable is causing changes in another. Blue connections indicate positive relationships (variables increase together), while red connections show negative relationships (one increases as another decreases). Purple dotted connections represent multivariate relationships where groups of variables jointly cause changes in a target variable. Additionally, thicker connections represent stronger correlations, and larger nodes indicate variables that influence many others, making key physiological drivers immediately apparent.

The statistical significance of each relationship is measured by its F-statistic and p-value. An F-statistic of 1-4 suggests weak causality with minimal predictive power, while 4-10 indicates moderate evidence with meaningful relationships. Strong causal evidence appears with F-statistics of 10-30, while values above 30 represent extremely strong causal relationships with overwhelming statistical significance (p << 0.001) and major predictive power.

## Additional Visualizations 
PhysioNexus provides two additional visualization types to enhance understanding of causal relationships:
- (1) Causal Matrix: This heatmap visualization displays the strength of causal relationships between all variables in a matrix format. The color intensity represents the F-statistic value, with darker colors indicating stronger causal relationships. This visualization is particularly useful for identifying clusters of variables with similar causal profiles and quickly spotting the strongest relationships in the dataset.
- (2) Causal Flow Diagram (Alluvial/Sankey Diagram): This interactive HTML-based visualization shows the flow of causality between variables, with the width of connections proportional to the strength of causal influence. The interactive nature of this visualization allows for exploration by hovering over connections to see details and repositioning nodes to better understand complex causal structures. Additionally, it provides an intuitive way to understand:
    - Which variables are major causal "sources" (primarily influencing others)
    - Which variables are "sinks" (primarily being influenced by others)
    - The relative strength of different causal pathways
    - The overall flow of causality through the physiological system

## Network Metrics and Their Interpretation
PhysioNexus provides several metrics that reveal different aspects of the physiological system's structure:
- Out-degree reveals a variable's influence by counting how many other variables it affects. High out-degree variables act as system "drivers" or control points that regulate multiple physiological processes.
- In-degree shows how responsive a variable is by counting how many variables affect it. High in-degree variables represent integration points where multiple physiological signals converge.
- Degree centrality combines both measures to identify the most connected variables overall, highlighting the central parameters in the physiological network regardless of direction.
- Betweenness centrality identifies mediator variables that frequently appear on paths between other variables. These "brokers" represent critical intermediate steps in physiological cascades, connecting different systems or processes.

# 🧬 Implementation: A DIY Guide To Using PhysioNexus

The most streamlined approach is to install PhysioNexus directly from GitHub using pip. This makes the function available for import in any of your Python environments without cluttering your code. Additionally, this method ensures you always have access to the latest version and keeps your analysis scripts clean and focused on your specific research questions.

You can install PhysioNexus as a Python package directly from Github using the code below:

```python
# Install the package directly from GitHub (Remove ! when running from Bash command line. Keep ! when running from notebook environment)
!pip install git+https://github.com/evanpeikon/PhysioNexus.git 

# Import and use
from physionexus import PhysioNexus
```
### Example 1: Basic Linear Causality Analysis

```python
# Install the PhysioNexus package directly from GitHub
!pip install git+https://github.com/evanpeikon/PhysioNexus.git

# Import PhysioNexus
from physionexus import PhysioNexus

# Load your data
data = pd.read_csv('Path to your CSV file', header=0)
data.dropna(inplace=True) # Remove rows / columns with missing values 

# Run PhysioNexus with linear causality testing
G, causal_df = PhysioNexus(
    data=data,  
    exclude_cols=['Time[s]', 'Time[hh:mm:ss]'],   # Replace with your non-feature columns
    corr_threshold=0.7,                           # Correlation threshold for considering relationships
    f_stat_threshold=10,                          # F-statistic threshold for Granger causality
    p_value_threshold=0.05,                       # P-value threshold for statistical significance
    max_lag=3,                                    # Maximum lag to consider for Granger causality
    output_dir=None)                              # Output directory to store results


# Optional: Display causal_df dataframe (uncomment line below to view)
# causal_df.head()
```

### Example 2: Multivariate Causality Analysis
```python
# Install the PhysioNexus package directly from GitHub
!pip install git+https://github.com/evanpeikon/PhysioNexus.git

# Import PhysioNexus
from physionexus import PhysioNexus

# Load your data
data = pd.read_csv('Path to your CSV file', header=0)
data.dropna(inplace=True) # Remove rows / columns with missing values 

# Define multivariate groups to test
# This example tests whether heart rate, muscle oxygenation, and VO2 jointly cause changes in lactate
multivariate_groups = {
    'Lactate': ['Heart_Rate', 'SmO2', 'VO2']} # You can add additional multivariate tests

# Run PhysioNexus with multivariate causality testing
G, causal_df = PhysioNexus(
    data=data,  
    exclude_cols=['Time[s]', 'Time[hh:mm:ss]'],   # Replace with your non-feature columns
    corr_threshold=0.7,                           # Correlation threshold for considering relationships
    f_stat_threshold=10,                           # F-statistic threshold for Granger causality
    p_value_threshold=0.05,                       # P-value threshold for statistical significance
    max_lag=3,                                    # Maximum lag to consider for Granger causality
    multivariate_groups=multivariate_groups,      # Define groups for multivariate testing
    output_dir=None)                              # Output directory to store results

# Optional: Display causal_df dataframe (uncomment line below to view)
# causal_df.head()
```


# 🧬 Elite Athlete Case Study
To demonstrate the tool's capabilities, we'll apply it to a case study using data from an athlete performing a ramp incremental exercise test to exhaustion. In this test, the athlete 
exercised at a progressively increasing intensity level until they could no longer continue, allowing us to observe the full spectrum of physiological responses from rest to maximal exertion. The data from this test was loaded into a pandas DataFrame and contains time series measurements for several physiological metrics:

- VO2 (oxygen consumption)
- Heart rate
- Heart rate variability 
- Power output
- Cycle rate
- Muscle recruitment (EMG)
- Respiratory rate 
- Tidal volume 
- Ventilatory exchange 
- Cerebral oxygenation
- Heart rate variability
- Muscle oxygenation
- Skin temperature
- Blood lactate

> Note: This case study assumes you've installed PhysioNexus using Method 1, from the implementation section above, though the functionality is identical with either implementation approach.

```python
# Install the PhysioNexus package directly from GitHub
!pip install git+https://github.com/evanpeikon/PhysioNexus.git

# Import PhysioNexus
from physionexus import PhysioNexus

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
PhysioNexus offers powerful analytical capabilities across human performance, research, and clinical domains by revealing complex physiological relationships through causal network analysis. Below i've outlined three areas where PhysioNexus is currently being employed:

- (1) Exercise Physiology Modeling: PhysioNexus is currently being used by professional sports teams and human performance groups within the DoD to map cause-effect relationships in time series data from asseesments and identify physiological drivers and response networks specific to individuals. Using this tool, coaches and human performance specialists can develop truly personalized training approaches based on these unique causal profiles, targeting the most influential variables for maximum impact. 
- (2) Training Effect Quantification: The causal network approach allows practitioners to quantify that X amount of training causes Y% change in target variables by determining edge weights in the network. By comparing causal networks before and after interventions (training programs, nutritional strategies, etc.), users can evaluate whether these interventions fundamentally alter physiological regulation. Perhaps most valuable is the ability to detect early warning signs of fatigue or overtraining through subtle changes in network structure before traditional markers appear, offering a novel monitoring system with predictive capabilities.
- (3) Environmental Adaptation Analysis: PhysioNexus enables powerful condition-specific comparisons between different environments or states. Researchers can examine how causal relationships transform when comparing high versus low altitude, earth versus space environments, or healthy versus pathological conditions. The enhanced visualizations, as of version 1.0.0, particularly the causal flow diagram, clearly illustrate which physiological systems function as central hubs in different scenarios and how regulatory mechanisms adapt to environmental challenges. This approach can reveal critical insights into physiological adaptation mechanisms that might be missed by conventional analysis methods.

## Current Limitations
While PhysioNexus offers significant analytical power, several limitations should be considered:

- (1) Linear Assumptions: The current implementation focuses on linear Granger causality and linear multivariate testing, which may not fully capture complex physiological relationships that often involve nonlinear dynamics. For instance, the relationship between heart rate variability and respiratory patterns typically follows nonlinear patterns that the current model might oversimplify.
- (2) Temporal Sensitivity: The analysis is sensitive to the sampling rate of the data, and different causal relationships may emerge at different time scales, potentially missing important connections that operate on very fast or slow timescales. For example, when analyzing muscle oxygenation data, sampling at 1Hz vs. 10Hz vs. 100Hz can yield substantially different causal networks. 
- (3) Hidden Variables: Unmeasured variables could drive apparent causal relationships between measured variables, creating misleading connections in the network.
- (4) Stationarity Assumption: The underlying statistical methods assume that the relationships between variables remain constant over time, which may not hold during dynamic physiological processes like exercise where regulation strategies can shift. For example, during incremental exercise tests, I've observed that causal relationships between VO2 and muscle oxygenation can subtely change above critical power, violating the stationarity assumption. 

## Extensions and Future Work
Several extensions could enhance PhysioNexus's capabilities and address current limitations:

- (1) Nonlinear Analysis: Implementing nonlinear causality measures using transfer entropy would better capture complex interactions in physiological systems that don't follow linear patterns. This would extend the tool's ability to detect subtle but important causal relationships that current methods might miss.
- (2) Multi-Scale Analysis: Developing functionality to analyze causality at different temporal resolutions would identify relationships that operate across various timescales, providing a more complete understanding of fast-acting and slow-developing physiological responses.
- (3) Differential Equation Modeling: Transforming causal networks into systems of ordinary differential equations would enable simulation and prediction of physiological responses. This would allow users to test hypothetical scenarios before implementation, creating a virtual physiological testing ground for optimizing interventions.
- (4) Dynamic Network Analysis: Extending the tool to examine how causal networks evolve during different exercise phases would capture the dynamic nature of physiological regulation during changing conditions, revealing how control mechanisms shift as the body responds to increasing demands.
- (5) Standardized Protocols: Developing protocols for comparing causal networks across different conditions would strengthen the tool's utility for understanding adaptive physiological responses and establish methodological consistency for research applications.

# 🧬 Contributing and Support
PhysioNexus is an open-source project and welcomes contributions from the community. If you encounter issues, have suggestions for improvements, or would like to contribute to the project, feel free to reach out: evanpeikon@gmail.com.
