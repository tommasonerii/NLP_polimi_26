import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

def main():
    # Resolve paths
    src_dir = Path(__file__).parent.resolve()
    project_root = src_dir.parent.parent
    log_path = project_root / "logs" / "simplewiki_tfidf_no_llm_all_competitions.csv"
    figures_dir = project_root / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        print(f"Error: Log file not found at {log_path}")
        sys.exit(1)

    print(f"Loading TF-IDF logs from {log_path.name}...")
    df = pd.read_csv(log_path)
    
    if df.empty:
        print("The log file is empty.")
        sys.exit(0)

    # Calculate summary metrics
    summary = df.groupby(['competition_id', 'competition_name']).agg(
        total_questions=('question_id', 'count'),
        correct_answers=('correct', 'sum'),
        max_level_reached=('level', 'max'),
        avg_latency_sec=('total_latency_seconds', 'mean')
    ).reset_index()
    
    summary['accuracy'] = summary['correct_answers'] / summary['total_questions']
    
    print("\n--- TF-IDF Strategy Summary ---")
    print(summary.to_string(index=False))
    
    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary, x='competition_name', y='accuracy')
    plt.title('TF-IDF Accuracy per Competition')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.tight_layout()
    acc_plot_path = figures_dir / "tfidf_accuracy.png"
    plt.savefig(acc_plot_path)
    print(f"\nSaved accuracy plot to {acc_plot_path}")
    
    # Plot Latency
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='competition_name', y='total_latency_seconds')
    plt.title('TF-IDF Response Latency per Competition')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    lat_plot_path = figures_dir / "tfidf_latency.png"
    plt.savefig(lat_plot_path)
    print(f"Saved latency plot to {lat_plot_path}")

if __name__ == "__main__":
    main()
