# 🧬 PhysioNexus: Mapping Cause and Effect In Time-Series Physiological Data
The evolution of physiological monitoring has followed a predictable path. We started with single metrics—heart rate monitors strapped to people’s chests—and gradually added more sensors, more data streams, and more complexity. The industry's solution to better physiological insights has consistently been "more data." But after a decade working with Olympians, professional athletes and teams, and human performance groups within the DoD, I've witnessed this approach reach its logical endpoint: a deluge of disconnected metrics that fail to reveal how our bodies actually function as integrated systems.

The fundamental limitation isn't in our measurement capabilities — after all, we can simultaneously track dozens of biometrics with nearly perfect precision. Rather, the limitation is our analytical framework. It's as if we've built one of the most advanced microscopes but are looking through it with one eye closed. For example, when an elite cyclist’s muscle oxygenation plummets during a climb traditional analysis can tell us that this correlates with rising blood lactate and heart rate— but correlation isn't causation. Which variable is driving the cascade? Which is merely responding?

This analytical dead-end led to the development of PhysioNexus, an open-source tool that moves beyond correlation to map causal networks in physiological data. PhysioNexus draws on bioinformatics techniques similar to those used for analyzing gene co-expression and protein-protein interaction networks, but adapts these approaches specifically for time-series physiological data. The approach uses Granger causality, which is a framework that defines causation through predictive power— If knowing the history of variable A helps us predict future values of variable B (beyond what past values of B alone can tell us), then A “Granger-causes” B. This directional relationship carries important information that correlation analysis misses entirely.

Consider a concrete example: In traditional analysis, seeing that respiratory rate rises as muscle oxygenation falls simply establishes correlation (in this case negative correlation). PhyioNexus, however, can determine that in a specific individual muscle oxygenation changes consistently precede and predict respiratory rate changes, suggesting a causal relationship where local tissue-level changes drive systemic cardiorespiratory responses— not the other way around. This distinction fundamentally changes how we understand and act on the data.

What makes physiological systems particularly challenging to analyze is their circular, interconnected nature. The body doesn't operate in neat linear pathways but through complex feedback loops— like one giant recursive algorithm. For example, When muscle oxygenation drops, ventilation increases, which affects blood pH, which influences oxygen binding affinity, which circles back to muscle oxygenation. PhysioNexus navigates this complexity by quantifying and visualizing the strength and direction of these interconnected relationships through multiple visualization approaches, including interactive network graphs, causal matrix heatmaps, and causal flow diagrams. Additionally, PhysioNexus' multivariate testing capabilities allow researchers to detect when combinations of variables jointly influence a target variable - an essential feature for understanding complex physiological systems where multiple signals might work together to drive changes in other systems.

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

<img width="350" alt="Screenshot 2025-04-26 at 7 49 57 AM" src="https://github.com/user-attachments/assets/bb09ef59-b414-4de4-9b83-406efcb37431" />

The network visualization transforms complex physiological relationships into an intuitive visual map. Single variables (ex, heart rate) are represented as blue nodes, and grouped variables (ex, heart rate, muscle oxygenation, and blood lactate) are represented as green nodes. Arrows between variables show the direction of influence—which variable is causing changes in another. Blue connections indicate positive relationships (variables increase together), while red connections show negative relationships (one increases as another decreases). Purple dotted connections represent multivariate relationships where groups of variables jointly cause changes in a target variable. Additionally, thicker connections represent stronger correlations, and larger nodes indicate variables that influence many others, making key physiological drivers immediately apparent.

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

# 🧬 Human Performance Case Study
## Background
To demonstrate PhysioNexus' capabilities, I analyzed data from an elite cyclist performing a ramp incremental exercise test to exhaustion. The test started at 150 watts on an indoor bike trainer with power increasing by 25 watts every 4 minutes until the athlete could no longer maintain the required power output. This protocol allowed us to observe the full spectrum of physiological responses from rest to maximal exertion.

The data included comprehensive physiological measurements including VO2 (ml/kg/min), heart rate (bpm), heart rate variability (ms), power (watts), cycle rate (rpm), muscle recruitment (mV), respiratory rate (breath/minute), tidal volume (L), ventilatory exchange (L/min), cerebral oxygenation (%), blood oxygenation (%), muscle oxygenation (%), skin temperature (C), and blood lactate (mmol). 

## Implementation
PhysioNexus was implemented with multivariate testing to analyze both direct causal relationships and the combined effects of variable groups:

```python
# Install the PhysioNexus package directly from GitHub
!pip install git+https://github.com/evanpeikon/PhysioNexus.git

# Import PhysioNexus
from physionexus import PhysioNexus

# Load Data
def load_data(path):
  data = pd.read_csv(path, header=0)
  data.dropna(inplace=True)
  return data

data = load_data('/content/drive/MyDrive/PhysioNexus/Test2.csv') 

# Define multivariate groups to test
multivariate_groups = {
    'SmO2' : ['Ventilatory_Exchange', 'HR[bpm]'],
    'HR[bpm]' : ['Ventilatory_Exchange', 'SmO2'],
    'VO2[mL/kg/min]': ['SmO2', 'HR[bpm]'],
    'HR[bpm]' : ['SmO2', 'VO2[mL/kg/min]'],
    'Ventilatory_Exchange': ['SmO2', 'VO2[mL/kg/min]'],
    'Ventilatory_Exchange': ['HR[bpm]', 'SmO2'],
    'Ventilatory_Exchange': ['HR[bpm]', 'VO2[mL/kg/min]}

# Run PhysioNexus with multivariate causality testing
G, causal_df = PhysioNexus(
    data=data,  
    exclude_cols=['Time[s]', 'Time[hh:mm:ss]'],   # Remove non-feature columns
    corr_threshold=0.7,                           # Correlation threshold for considering relationships
    f_stat_threshold=10,                          # F-statistic threshold for Granger causality
    p_value_threshold=0.05,                       # P-value threshold for statistical significance
    max_lag=3,                                    # Maximum lag to consider for Granger causality
    multivariate_groups=multivariate_groups,      # Define groups for multivariate testing
    output_dir=None)                              
```
The code above produced the following visualizations:

#### Figure 1: Causal Network

This directed graph below illustrated causal relationships between physiological variables, with blue arrows showing positive linear causality, red arrows indicating negative linear causality, and purple dotted lines representing multivariate causality. The network reveals central roles for SmO2, heart rate, and RR intervals with extensive connections, while also highlighting three multivariate relationships involving combinations of variables jointly influencing specific outcomes.

<img width="600" alt="Screenshot 2025-04-25 at 2 47 32 PM" src="https://github.com/user-attachments/assets/727178e3-5866-4c3e-9265-3729f772c4a0" />

#### Figure 2: Causal Strength Matrix

This heatmap visualization below displays the F-statistic values representing causal relationship strength between physiological variables during incremental exercise. Darker blue squares indicate stronger causal relationships, with the highest values (142.2) observed between heart rate and RR intervals, while the prominent influence of power output on variables like SmO2 and blood lactate is clearly visible.

<img width="500" alt="Screenshot 2025-04-25 at 2 48 05 PM" src="https://github.com/user-attachments/assets/9e2dda29-4d2d-42d7-af25-80c46dca107f" />

#### Figure 3: Causal Flow Diagram

This Sankey diagram below depicts the flow of causality through the physiological system, with node width and connecting flow thickness proportional to causal strength. The visualization shows how causal influence propagates from primary variables (heart rate, power, SmO2) through intermediary processes to terminal outcomes, while also highlighting three multivariate groups (shown in the top left) that exert joint causal effects on specific variables.

<img width="750" alt="Screenshot 2025-04-25 at 2 47 50 PM" src="https://github.com/user-attachments/assets/c9b60914-6a91-4f99-8831-e3ac7adb4ade" />

## Network Metrics and Key Insights 
### Causal Drivers (Out-degree)
The top 5 nodes by out-degree (causal influence) in our network are as follows:
- Muscle Oxygenation (SmO2): 10 outgoing connections
- Blood Lactate: 10 outgoing connections
- VO2: 8 outgoing connections
- Heart rate (HR): 6 outgoing connections
- Respiratory_Rate (RR): 6 outgoing connections

Key Insight - Peripheral Dominance: Unlike typical endurance athletes where heart rate is the dominant causal driver, this cyclist shows muscle oxygenation (SmO2) and blood lactate as the strongest causal drivers with 10 outgoing connections each. This reveals a peripherally-regulated performance profile where local muscle conditions drive systemic responses. For coaches, this means training interventions should prioritize improving local muscle adaptations such as capillarization, mitochondrial density, and lactate clearance capacity rather than focusing primarily on central cardiovascular improvements. Specifically, incorporating more work at lactate threshold (sweet spot training) and polarized intensity distribution would optimize this athlete's unique physiological profile.

### Network Centrality (Overall Importance)
The top 5 nodes by degree centrality (overall connection importance) in our network are:
- Muscle Oxygenation (SMO2): 1.1429
- Blood Lactate: 1.0714
- Respiratory Rate: 1.0000
- VO2: 0.9286
- Heart Rate: 0.7143

Key Insight - Metabolic Monitoring Priority: The centrality metrics definitively identify muscle oxygenation and blood lactate as the most important overall variables in this athlete's physiological network. This suggests implementing real-time muscle oxygenation monitoring during training would provide the most comprehensive insights into this athlete's physiological state. Since SmO2 can be continuously monitored non-invasively while lactate requires intermittent blood sampling, a NIRS-based SmO2 monitor would be the optimal practical solution for daily training guidance. Establishing training zones based on SmO2 thresholds rather than traditional heart rate or power zones would better align with this athlete's unique physiological regulation pattern.

### Information Flow Hubs (Betweenness Centrality):
- The op 5 nodes by betweenness centrality (information flow brokers) are:
- Muscle Oxygenation (SMO2): 0.1676
- Ventilatory Exchange: 0.1648
- VO2: 0.1480
- Blood Lactate: 0.1071
- Respiratory Rate: 0.0839

Key Insight - Critical Pathway Identification: Muscle oxygenation, ventilatory exchange, and oxygen consumption form the critical information flow pathway in this athlete's physiological network. These three variables together mediate how other physiological responses interact, suggesting they should be the primary targets for intervention and monitoring. For this athlete, training should be designed with a specific focus on improving the coordination between these systems. Practically, this means implementing "stacked intervals" where one physiological system is pre-fatigued before targeting another (e.g., respiratory pre-fatigue through restricted breathing warm-up before high-intensity intervals) to enhance the integration between these key regulatory hubs.

### Strongest Causal Relationships
- The top linear causal relationships by F-statistic in this network are as follows:
- Heart rate → Respiratory Rate: F=142.16, p=0.00000, correlation=-0.955
- Heart rate → Ventilatory_Exchange: F=103.26, p=0.00000, correlation=0.929
- Power → SmO2: F=100.67, p=0.00000, correlation=-0.613
- Power → Blood Lactate: F=100.67, p=0.00000, correlation=0.613

Key Insight - Direct Performance Levers: The strongest causal relationships reveal four primary "control knobs" that directly drive physiological responses: heart rate's effect on breathing (HR→RR, HR→Ventilatory Exchange), and power output's dual effect on muscle oxygenation and blood lactate. For the coach, this means power-based training should be precisely calibrated, as even small changes in power output will trigger substantial muscle oxygenation and lactate responses. This athlete requires very narrow, precisely-defined training zones compared to typical athletes. The data suggests establishing micro-zones with 10-20 watt increments rather than the standard ~50 watt zones would optimize training specificity and ensure targeted physiological adaptations.

### Multivariate Causality
Multivariate causal relationships revealed by this analysis include:
- Group(Ventilatory Exchange, Heart Rate) → SmO2: F=14.22, p=0.00000, lag=2
- Group(SmO2, Heart Rate) → VO2 : F=47.86, p=0.00000, lag=1
- Group(Heart Rate, VO2) → Ventilatory Exchange: F=21.64, p=0.00000, lag=1

Key Insight - Integrated Systems Approach: The multivariate analysis reveals sophisticated regulation where combinations of variables jointly influence outcomes in ways that single variables cannot. Most notably, muscle oxygenation appears to be jointly controlled by the combination of ventilation and heart rate rather than by either variable independently. This has profound training implications, indicating that isolated interventions targeting either cardiac output or breathing mechanics alone won't optimize this athlete's muscle oxygenation. Instead, combined interventions are required. Specifically, a respiratory muscle training program synchronized with cardiac interval training would more effectively improve this athlete's muscle oxygenation dynamics than either intervention alone, creating a multiplicative rather than additive training effect.

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
