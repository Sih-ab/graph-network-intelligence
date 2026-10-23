import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# Graph Network Intelligence - Streamlit dashboard

st.set_page_config(
    page_title="Graph Network Intelligence",
    page_icon="🔗",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "raw" / "email-Eu-core.txt.gz"
MODELS_DIR = BASE_DIR / "models"


@st.cache_data(show_spinner=False)
def load_edges(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=r"\s+",
        compression="gzip",
        header=None,
        names=["source", "target"],
    )


@st.cache_resource(show_spinner="Loading network...")
def load_graph(path: str) -> nx.Graph:
    edges = load_edges(path)
    graph = nx.from_pandas_edgelist(
        edges, source="source", target="target", create_using=nx.Graph()
    )
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def read_optional_result(filename: str):
    path = MODELS_DIR / filename
    if not path.exists():
        return None
    try:
        return load_csv(str(path))
    except Exception as exc:
        st.error(f"Could not read {filename}: {exc}")
        return None


def show_dataset_error():
    st.error(
        "Dataset not found. Expected file: "
        "`data/raw/email-Eu-core.txt.gz`. "
        "Please check the file location in your project."
    )


def show_network_metrics(graph: nx.Graph, four_columns=True):
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    density = nx.density(graph) if nodes > 1 else 0.0
    avg_degree = sum(dict(graph.degree()).values()) / nodes if nodes else 0.0

    if four_columns:
        cols = st.columns(4)
        cols[0].metric("🔵 Total Nodes", f"{nodes:,}")
        cols[1].metric("🔗 Total Edges", f"{edges:,}")
        cols[2].metric("📈 Network Density", f"{density:.4f}")
        cols[3].metric("🔄 Average Degree", f"{avg_degree:.2f}")
    else:
        cols = st.columns(3)
        cols[0].metric("🔵 Total Nodes", f"{nodes:,}")
        cols[1].metric("🔗 Total Edges", f"{edges:,}")
        cols[2].metric("📈 Network Density", f"{density:.4f}")


def show_comparison_chart(df: pd.DataFrame, metric: str, title: str):
    if "Method" not in df.columns or metric not in df.columns:
        st.info(f"Column(s) needed for {title} were not found in the CSV.")
        return
    chart_df = df[["Method", metric]].copy()
    chart_df[metric] = pd.to_numeric(chart_df[metric], errors="coerce")
    chart_df = chart_df.dropna()
    if not chart_df.empty:
        st.subheader(title)
        st.bar_chart(chart_df.set_index("Method"))


# Sidebar navigation is created BEFORE page-specific content.
st.sidebar.title("🔗 Graph Intelligence")
st.sidebar.caption("Network Analysis & Machine Learning")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    [
        "Dashboard",
        "Graph Exploration",
        "Community Detection",
        "Link Prediction",
        "Model Comparison",
        "Project Information",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Email-Eu-core network dataset")

# Dashboard
if page == "Dashboard":
    st.title("🔗 Graph Network Intelligence")
    st.subheader("Explore Networks, Communities, and Link Prediction")
    st.write(
        "Welcome! Explore network structure, discover communities, "
        "and compare link-prediction methods in this interactive dashboard."
    )
    st.divider()

    st.header("📊 Network Statistics")
    if DATASET_PATH.exists():
        try:
            graph = load_graph(str(DATASET_PATH))
            show_network_metrics(graph, four_columns=True)
            st.success("Network statistics loaded successfully.")
        except Exception as exc:
            st.error(f"Could not load the dataset: {exc}")
    else:
        show_dataset_error()

    st.divider()
    st.header("👥 Community Detection Visualization")
    image_path = MODELS_DIR / "community_visualization.png"
    if image_path.exists():
        st.image(
            str(image_path),
            caption="Louvain Community Detection",
            use_container_width=True,
        )
    else:
        st.info("Community visualization image was not found.")

    comparison = read_optional_result("same_test_model_comparison.csv")
    if comparison is not None:
        st.divider()
        st.header("🏆 Model Performance Overview")
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        show_comparison_chart(comparison, "ROC-AUC", "ROC-AUC by Model")


# Graph Exploration
elif page == "Graph Exploration":
    st.title("🌐 Graph Exploration")
    st.write("Explore the size, connectivity, and degree distribution of the network.")

    if not DATASET_PATH.exists():
        show_dataset_error()
    else:
        try:
            graph = load_graph(str(DATASET_PATH))
            show_network_metrics(graph, four_columns=False)

            st.divider()
            st.subheader("📊 Degree Distribution")
            degree_counts = pd.Series(
                [degree for _, degree in graph.degree()], name="Degree"
            ).value_counts().sort_index()
            degree_counts.index.name = "Degree"
            degree_counts.name = "Number of Nodes"
            st.bar_chart(degree_counts)

            st.divider()
            st.subheader("🏆 Top 10 Nodes by Degree")
            top_nodes = sorted(
                graph.degree(), key=lambda item: item[1], reverse=True
            )[:10]
            top_df = pd.DataFrame(top_nodes, columns=["Node ID", "Degree"])
            st.dataframe(top_df, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("🕸️ Network Visualization")
            max_nodes = min(200, graph.number_of_nodes())
            sample_size = st.slider(
                "Number of high-degree nodes to display",
                min_value=min(20, max_nodes),
                max_value=max_nodes,
                value=min(80, max_nodes),
                step=10 if max_nodes >= 30 else 1,
            )
            chosen = [
                node for node, _ in sorted(
                    graph.degree(), key=lambda item: item[1], reverse=True
                )[:sample_size]
            ]
            subgraph = graph.subgraph(chosen)
            fig, ax = plt.subplots(figsize=(10, 7))
            positions = nx.spring_layout(subgraph, seed=42)
            nx.draw_networkx(
                subgraph,
                pos=positions,
                ax=ax,
                node_size=35,
                with_labels=False,
                edge_color="gray",
                width=0.5,
                alpha=0.7,
            )
            ax.set_title("Sample of the Email Communication Network")
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)
        except Exception as exc:
            st.error(f"Graph exploration failed: {exc}")


# Community Detection
elif page == "Community Detection":
    st.title("👥 Community Detection")
    st.write(
        "Find groups of nodes that are more densely connected to one another."
    )

    image_path = MODELS_DIR / "community_visualization.png"
    if image_path.exists():
        st.image(
            str(image_path),
            caption="Saved Louvain Community Detection Visualization",
            use_container_width=True,
        )
    else:
        st.info("Community visualization image was not found.")

    st.divider()
    if st.button("Calculate Community Statistics", type="primary"):
        if not DATASET_PATH.exists():
            show_dataset_error()
        else:
            try:
                graph = load_graph(str(DATASET_PATH))
                with st.spinner("Calculating communities..."):
                    communities = nx.community.louvain_communities(graph, seed=42)
                sizes = sorted(
                    [len(community) for community in communities], reverse=True
                )

                c1, c2 = st.columns(2)
                c1.metric("Number of Communities", len(communities))
                c2.metric("Largest Community", max(sizes) if sizes else 0)

                size_df = pd.DataFrame(
                    {
                        "Community ID": range(1, len(sizes) + 1),
                        "Number of Nodes": sizes,
                    }
                )
                st.subheader("Community Sizes")
                st.dataframe(size_df, use_container_width=True, hide_index=True)
                st.bar_chart(size_df.set_index("Community ID"))
            except Exception as exc:
                st.error(f"Community detection failed: {exc}")


# Link Prediction
elif page == "Link Prediction":
    st.title("🔗 Link Prediction")
    st.write("Review the saved Node2Vec + Logistic Regression results.")

    results = read_optional_result("node2vec_link_prediction_results.csv")
    if results is None:
        st.warning(
            "Results file not found: "
            "`models/node2vec_link_prediction_results.csv`."
        )
    else:
        st.subheader("Link Prediction Results")
        st.dataframe(results, use_container_width=True, hide_index=True)

        numeric_cols = results.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            metric = st.selectbox("Choose a metric to chart", numeric_cols)
            values = pd.to_numeric(results[metric], errors="coerce").dropna()
            if not values.empty:
                st.subheader(metric)
                st.bar_chart(values)

        st.download_button(
            "⬇️ Download Link Prediction Results",
            data=results.to_csv(index=False),
            file_name="node2vec_link_prediction_results.csv",
            mime="text/csv",
        )

    st.divider()
    st.subheader("How Link Prediction Works")
    st.markdown(
        """
        1. Split known edges into training and testing sets.
        2. Sample node pairs without an edge as negative examples.
        3. Train Node2Vec embeddings using the training graph.
        4. Build edge features from node embeddings.
        5. Train Logistic Regression and evaluate its predictions.
        """
    )


# Model Comparison
elif page == "Model Comparison":
    st.title("📊 Link Prediction Model Comparison")
    st.write("Compare methods evaluated on the same test dataset.")

    comparison = read_optional_result("same_test_model_comparison.csv")
    if comparison is None:
        st.warning(
            "Comparison file not found: "
            "`models/same_test_model_comparison.csv`."
        )
    else:
        st.subheader("Model Comparison Table")
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        show_comparison_chart(comparison, "ROC-AUC", "ROC-AUC Comparison")
        show_comparison_chart(
            comparison, "Average Precision", "Average Precision Comparison"
        )
        st.download_button(
            "⬇️ Download Model Comparison CSV",
            data=comparison.to_csv(index=False),
            file_name="same_test_model_comparison.csv",
            mime="text/csv",
        )


# Project Information
elif page == "Project Information":
    st.title("ℹ️ Project Information")
    st.write(
        "Graph Network Intelligence is a data science project for exploring "
        "network structure, detecting communities, and comparing link-prediction methods."
    )

    st.divider()
    st.subheader("📁 Dataset")
    st.markdown(
        """
        **Email-Eu-core**

        - Nodes represent participants in the network.
        - Edges represent communication connections.
        - The dashboard treats the graph as undirected.
        """
    )

    st.divider()
    st.subheader("🛠️ Technologies")
    st.markdown(
        """
        - Python
        - Pandas
        - NetworkX
        - Matplotlib
        - Scikit-learn
        - Node2Vec / Gensim
        - Streamlit
        """
    )

    st.divider()
    st.subheader("🧠 Methods")
    st.markdown(
        """
        - Graph exploration and degree analysis
        - Louvain community detection
        - Common Neighbors
        - Jaccard Coefficient
        - Preferential Attachment
        - Node2Vec embeddings
        - Logistic Regression
        """
    )

    st.divider()
    st.subheader("🎯 Project Objectives")
    st.markdown(
        """
        1. Explore a real-world communication network.
        2. Identify communities within the network.
        3. Predict possible links between nodes.
        4. Compare link-prediction methods using evaluation metrics.
        5. Present the findings in an interactive dashboard.
        """
    )

    st.divider()
    st.subheader("📂 Project Structure")
    st.code(
        """Graph Network Intelligence/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   └── raw/
│       └── email-Eu-core.txt.gz
├── models/
│   ├── community_visualization.png
│   ├── node2vec_embeddings.csv
│   ├── node2vec_link_prediction_results.csv
│   └── same_test_model_comparison.csv
└── notebooks/
    └── 01_graph_exploration.ipynb""",
        language="text",
    )

    st.info(
        "This dashboard displays saved results. It does not retrain the "
        "Node2Vec model whenever a page is opened."
    )


st.divider()
st.caption("Graph Network Intelligence • Network Analysis & Machine Learning")
