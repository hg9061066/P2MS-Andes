import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_federation_advantage():
    print("--- Generating Final Performance Visualization ---")
    df = pd.read_csv("data/processed/final_metrics.csv")
    
    # Set up the plot
    labels = df['Node'].apply(lambda x: x.replace('Node_', '').replace('_', ' '))
    local_f1 = df['Local_F1']
    fed_f1 = df['Federated_F1']
    
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, local_f1, width, label='Local-Only (Siloed)', color='#95a5a6')
    rects2 = ax.bar(x + width/2, fed_f1, width, label='P²MS-Andes (Federated)', color='#2ecc71')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Macro F1-Score')
    ax.set_title('Federation Advantage: Local vs. Federated Risk Forecasting')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add boost annotations
    for i, boost in enumerate(df['Boost']):
        color = 'green' if boost > 0 else 'red'
        ax.annotate(f"{'+' if boost > 0 else ''}{boost}",
                    xy=(i + width/2, fed_f1[i]),
                    xytext=(0, 3), # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color=color, fontweight='bold')

    plt.ylim(0, 0.5) # Adjust based on your F1 range
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("data/processed/federation_advantage.png")
    print("-> Final Figure saved to: data/processed/federation_advantage.png")

if __name__ == "__main__":
    plot_federation_advantage()