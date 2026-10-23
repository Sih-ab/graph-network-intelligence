# Graph Network Intelligence

## Project Overview

Graph Network Intelligence is a graph machine learning project that analyzes an email communication network using graph algorithms and node embeddings.

The project explores network structure, predicts potential links between nodes, detects communities, and compares different link prediction methods.

## Objectives

* Explore the structure of an email communication network.
* Generate node embeddings using Node2Vec.
* Predict potential links between nodes.
* Compare traditional graph-based heuristics with a machine learning approach.
* Detect communities using the Louvain algorithm.
* Visualize the network and model evaluation results.

## Dataset

**Dataset:** Email-Eu-core Network

The dataset represents an email communication network. Nodes represent network members, and edges represent email relationships.

## Technologies Used

* Python
* Pandas
* NumPy
* NetworkX
* Node2Vec
* Gensim
* Scikit-learn
* Matplotlib
* Jupyter Notebook

## Methods

### 1. Graph Exploration

The dataset is loaded into an undirected NetworkX graph. Self-loops are removed before the main graph analysis.

### 2. Node Embeddings

Node2Vec generates a 64-dimensional embedding for each of the 1,005 nodes.

### 3. Link Prediction

The project compares four approaches:

* Common Neighbors
* Jaccard Coefficient
* Preferential Attachment
* Node2Vec with Logistic Regression

For the machine learning approach, the element-wise product of two node embeddings is used to construct pairwise features.

### 4. Community Detection

The Louvain algorithm identifies communities in the email network.

## Model Evaluation

The models were evaluated using ROC-AUC and Average Precision.

### Results on the Same Test Set

| Method                         | ROC-AUC | Average Precision |
| ------------------------------ | ------: | ----------------: |
| Common Neighbors               |  0.9422 |            0.9368 |
| Jaccard Coefficient            |  0.9321 |            0.9311 |
| Preferential Attachment        |  0.8716 |            0.8698 |
| Node2Vec + Logistic Regression |  0.7644 |            0.7580 |

These results indicate that Common Neighbors achieved the highest scores among the four methods in this evaluation setup.

## Visualizations

The `models/` directory contains:

* `node2vec_performance.png` — Node2Vec evaluation metrics.
* `model_comparison.png` — Initial model comparison.
* `same_test_model_comparison.png` — Comparison using the same test set.
* `community_visualization.png` — Network communities visualized by color.

## Project Structure

```text
Graph Network Intelligence/
├── app.py
├── data/
├── models/
│   ├── community_visualization.png
│   ├── link_prediction_results.csv
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── node2vec_embeddings.csv
│   ├── node2vec_link_prediction_results.csv
│   ├── node2vec_performance.png
│   ├── same_test_model_comparison.csv
│   └── same_test_model_comparison.png
├── notebooks/
│   └── 01_graph_exploration.ipynb
├── src/
├── README.md
└── requirements.txt
```

## How to Run

### 1. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Dashboard

Start the interactive Graph Network Intelligence dashboard:

```bash
streamlit run app.py
```

The dashboard will open in your web browser. You can explore the graph, inspect communities, view link prediction results, and compare models.

### 3. Run the Jupyter Notebook

Open the following notebook in VS Code or Jupyter:

`notebooks/01_graph_exploration.ipynb`

Select the Python environment containing the project dependencies and run the notebook cells in order.

### 4. Dataset Requirements

Ensure that the dataset is available at the expected location:

`data/raw/email-Eu-core.txt.gz`

## Limitations

* Link prediction performance depends on the dataset, graph representation, embedding parameters, and negative sampling strategy.
* The current comparison uses a particular train/test split and sampled negative pairs.
* Further experiments are needed to evaluate alternative embedding parameters and additional graph machine learning methods.

## Future Improvements

* Tune Node2Vec hyperparameters.
* Explore Graph Neural Networks.
* Improve link prediction feature engineering.
* Add interactive network visualizations.
* Develop an interactive application for exploring the graph.

## Conclusion

This project demonstrates how graph analysis, node embeddings, link prediction, and community detection can be combined to study an email communication network.
