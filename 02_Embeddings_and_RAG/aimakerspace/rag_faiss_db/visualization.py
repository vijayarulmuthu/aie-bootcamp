import sqlite3
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json

class RAGVisualizer:
    def __init__(self, db_path: str = "rag_faiss_db/results/results.db"):
        """Initialize the visualizer with database path"""
        self.db_path = db_path
        self._validate_db()
        # Define consistent colors for each search method
        self.method_colors = {
            "flat": "#1f77b4",  # blue
            "lsh": "#ff7f0e",  # orange
            "hnsw": "#2ca02c",  # green
            "ivf": "#d62728"  # red
        }

    def _validate_db(self):
        """Validate database connection and structure"""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file not found: {self.db_path}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rag_results_faiss_db'")
            if not cursor.fetchone():
                raise ValueError("Database does not contain required 'rag_results_faiss_db' table")
            conn.close()
        except sqlite3.Error as e:
            raise ConnectionError(f"Failed to connect to database: {str(e)}")

    def _load_data(self) -> pd.DataFrame:
        """Load data from SQLite database into pandas DataFrame"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query("""
                SELECT 
                    timestamp,
                    query,
                    search_method,
                    search_time,
                    error
                FROM rag_results_faiss_db
            """, conn)
            conn.close()
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
        except Exception as e:
            raise ValueError(f"Failed to load data: {str(e)}")

    def visualize_query_results(self, output_dir: str = "rag_faiss_db/results") -> str:
        """
        Create visualizations for query results and save to HTML file.
        
        Args:
            output_dir: Directory to save the HTML file
            
        Returns:
            str: Path to the generated HTML file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        df = self._load_data()
        
        if df.empty:
            print("No data found in the database.")
            return ""
        
        # Create HTML file path
        html_file = os.path.join(output_dir, "query_analysis.html")
        
        # Group by query
        unique_queries = df["query"].unique()
        
        # Create a single figure with subplots for each query
        fig = make_subplots(
            rows=len(unique_queries), cols=1,
            subplot_titles=[f"Query: {query}" for query in unique_queries],
            vertical_spacing=0.1
        )
        
        for i, query in enumerate(unique_queries, 1):
            query_df = df[df["query"] == query]
            
            # Create bar chart for search times with consistent colors per method
            for method in query_df["search_method"].unique():
                method_df = query_df[query_df["search_method"] == method]
                fig.add_trace(
                    go.Bar(
                        x=[method],
                        y=method_df["search_time"],
                        text=method_df["search_time"].round(2),
                        textposition="auto",
                        name=method,
                        marker_color=self.method_colors[method],
                        showlegend=(i == 1)  # Only show legend for first subplot
                    ),
                    row=i, col=1
                )
            
            # Add error information if present
            error_rows = query_df[query_df["error"].notna()]
            for _, row in error_rows.iterrows():
                fig.add_annotation(
                    text=f"Error ({row['search_method']}): {row['error']}",
                    xref=f"x{i}",
                    yref=f"y{i}",
                    x=row["search_method"],
                    y=row["search_time"],
                    showarrow=True,
                    arrowhead=1,
                    font=dict(color="red"),
                    row=i, col=1
                )
            
            # Update axes labels
            fig.update_xaxes(title_text="Search Method", row=i, col=1)
            fig.update_yaxes(title_text="Search Time (seconds)", row=i, col=1)
        
        # Update layout
        fig.update_layout(
            height=300 * len(unique_queries),
            title_text="Search Time Comparison by Method",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            barmode="group"
        )
        
        # Save to HTML
        fig.write_html(html_file)
        print(f"Query analysis saved to {html_file}")
        
        return html_file

if __name__ == "__main__":
    # Example usage
    visualizer = RAGVisualizer()
    visualizer.visualize_query_results() 