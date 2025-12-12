import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import zscore

# Set style
sns.set(style="whitegrid")

# Create output directory
output_dir = "figures/k3_summary"
os.makedirs(output_dir, exist_ok=True)

# Load data
cluster_df = pd.read_csv("Embedding - K=3/cluster_assignments_k3.csv")
desc_df = pd.read_excel("Embedding - K=3/Descriptive_Table_By_Cluster_UniqueCells.xlsx")
anova_df = pd.read_excel("Embedding - K=3/ANOVA - OneWay.xlsx", skiprows=1)

# ================================
# Cell Group Distribution Plot
# ================================
# Build unique IDs and compute percentages as before...
cluster_df['UniqueID'] = cluster_df['Experiment'].astype(str) + "_" + cluster_df['Parent'].astype(str)
group_counts = cluster_df.groupby(['Treatment', 'Cluster'])['UniqueID'] \
                         .nunique().unstack(fill_value=0)
group_counts = group_counts.reindex(
    sorted(group_counts.index, key=lambda x: (x != "CON0", x))
)
group_percent = group_counts.div(group_counts.sum(axis=1), axis=0) * 100

colors = ['red', 'blue', 'green']
# Plot with custom size and no x-axis label
ax = group_percent.plot(
    kind='bar',
    stacked=False,
    color=colors,
    figsize=(16, 5)   # Wider figure for better readability
)
ax.set_ylabel('Cell Distribution (%)')
ax.set_title('Cell Distribution Per Treatment')
ax.set_xlabel('')     # Remove x-axis label (no “Treatment” text)
ax.legend(
    labels=['Group 0', 'Group 1', 'Group 2'],
    loc='lower center',
    bbox_to_anchor=(0.5, -0.4),
    ncol=3,
    frameon=False
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='right')
plt.subplots_adjust(bottom=0.45)
plt.savefig(f"{output_dir}/Cell_Distribution_Per_Treatment.png", dpi=300)
plt.close()


# ================================
# Normalized Feature Ratios to G0
# ================================
features_df = desc_df.set_index('Cluster').T
features_df.columns = ['G0', 'G1', 'G2']

# Keep only *_Mean features (exclude Std, SE, CI, N_Cells etc.)
mean_features = [col for col in features_df.index if col.endswith('_Mean') and col != 'N_Cells']
features_df = features_df.loc[mean_features]

norm_df = pd.DataFrame()
norm_df['G0/G0'] = features_df['G0'].abs() / features_df['G0'].abs() * 100
norm_df['G1/G0'] = features_df['G1'].abs() / features_df['G0'].abs() * 100
norm_df['G2/G0'] = features_df['G2'].abs() / features_df['G0'].abs() * 100

norm_df = norm_df.reset_index().rename(columns={'index': 'Feature'})
norm_df['DisplayName'] = norm_df['Feature'].str.replace('_Mean$', '', regex=True)
norm_df.to_csv(f"{output_dir}/Normalized_Feature_Ratios.csv", index=False)

# Merge with ANOVA p-values
anova_df = anova_df.rename(columns={'Unnamed: 0': 'Feature', 'Unnamed: 5': 'p-value'})
merged_df = pd.merge(norm_df, anova_df[['Feature', 'p-value']], on='Feature', how='left')

# Categorize features explicitly
morph_features_raw = ['Area', 'Ellip_Ax_B_X','Ellip_Ax_B_Y','Ellip_Ax_C_X','Ellip_Ax_C_Y','EllipsoidAxisLengthB','EllipsoidAxisLengthC','Ellipticity_oblate','Ellipticity_prolate','Sphericity','Eccentricity']
kinetic_features_raw = ['Acceleration','Acceleration_OLD','Acceleration_X','Acceleration_Y','Confinement_Ratio','Directional_Change','Overall_Displacement','Displacement_From_Last_Id','Displacement2','Instantaneous_Speed','Instantaneous_Speed_OLD','Linearity_of_Forward_Progression','Mean_Curvilinear_Speed','Mean_Straight_Line_Speed','Current_MSD_1','Final_MSD_1','MSD_Linearity_R2_Score','MSD_Brownian_Motion_BIC_Score','MSD_Brownian_D','MSD_Directed_Motion_BIC_Score','MSD_Directed_D','MSD_Directed_v2','Total_Track_Displacement','Track_Displacement_X','Track_Displacement_Y','Velocity_X','Velocity_Y','Min_Distance','Velocity_Full_Width_Half_Maximum','Velocity_Time_of_Maximum_Height','Velocity_Maximum_Height','Velocity_Ending_Value','Velocity_Ending_Time','Velocity_Starting_Value','Velocity_Starting_Time']

morph_features = [f + '_Mean' for f in morph_features_raw]
kinetic_features = [f + '_Mean' for f in kinetic_features_raw]

merged_df['Category'] = merged_df['Feature'].apply(
    lambda x: 'Morphological' if x in morph_features else ('Kinetic' if x in kinetic_features else 'Other')
)

# ================================
# Plot Normalized Feature Ratios (Barplots)
# ================================
def plot_normalized_features(df, label, file_prefix):
    chunk_size = 18
    max_display = 1000  # clipping threshold
    num_chunks = int(np.ceil(len(df) / chunk_size))

    for i in range(num_chunks):
        chunk = df.iloc[i*chunk_size:(i+1)*chunk_size].copy()
        chunk_melted = chunk.melt(
            id_vars=['Feature', 'DisplayName'],
            value_vars=['G0/G0', 'G1/G0', 'G2/G0'],
            var_name='Group',
            value_name='Value'
        )

        plt.figure(figsize=(14, 6))
        ax = sns.barplot(
            data=chunk_melted,
            x='DisplayName',
            y='Value',
            hue='Group',
            palette=['red', 'blue', 'green']
        )

        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.ylabel('Normalized to Group 0 (%)')
        plt.ylim(0, max_display)
        plt.title(f'{label} - Part {i+1}')
        plt.legend(title='Group', ncol=3)

        # Add manual labels ONLY for bars > max_display
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if height > max_display:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        max_display - 50,  # slightly below the top
                        f'{height:.0f}%',
                        ha='center',
                        va='top',
                        fontsize=11,
                        color='white',
                        rotation=90
                    )

        plt.subplots_adjust(bottom=0.5)
        plt.savefig(f"{output_dir}/{file_prefix}_Part_{i+1}.png", dpi=300)
        plt.close()

    # --- Leave All-in-One Plot as-is ---
    full_melted = df.melt(
        id_vars=['Feature', 'DisplayName'],
        value_vars=['G0/G0', 'G1/G0', 'G2/G0'],
        var_name='Group',
        value_name='Value'
    )
    plt.figure(figsize=(22, 6))
    ax = sns.barplot(
        data=full_melted,
        x='DisplayName',
        y='Value',
        hue='Group',
        palette=['red', 'blue', 'green']
    )
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.ylabel('Normalized to Group 0 (%)')
    plt.title(f'{label}')
    plt.legend(title='Group', ncol=3)

    # No annotations here
    plt.subplots_adjust(bottom=0.5)
    plt.savefig(f"{output_dir}/{file_prefix}_All_Features.png", dpi=300)
    plt.close()


# Create the normalized ratio plots
plot_normalized_features(merged_df, 'Normalized Features - All', 'Normalized_All')
plot_normalized_features(merged_df[merged_df['Category'] == 'Morphological'], 'Normalized Morphological Features', 'Normalized_Morph')
plot_normalized_features(merged_df[merged_df['Category'] == 'Kinetic'], 'Normalized Kinetic Features', 'Normalized_Kinetic')

# === subplots for Kinetic and Morphological features ===
def plot_normalized_subplots(df, output_dir):
    """
    Create a figure with three vertically stacked subplots:
      1) First half of Kinetic features
      2) Second half of Kinetic features
      3) All Morphological features

    Each subplot uses its own Y-axis scale.
    A single legend (without a title) is placed above the main title.
    """
    os.makedirs(output_dir, exist_ok=True)
    max_display = 1000  # clipping threshold

    # Split DataFrame by category
    kin_df = df[df['Category'] == 'Kinetic'].copy()
    morph_df = df[df['Category'] == 'Morphological'].copy()

    num_kin = len(kin_df)
    half = num_kin // 2
    kin1 = kin_df.iloc[:half]
    kin2 = kin_df.iloc[half:]

    fig, axes = plt.subplots(3, 1, figsize=(20, 15))

    subsets = [
        (kin1, "Kinetic Features (Part 1)"),
        (kin2, "Kinetic Features (Part 2)"),
        (morph_df, "Morphological Features")
    ]

    for ax, (sub_df, title) in zip(axes, subsets):
        melted = sub_df.melt(
            id_vars=['Feature', 'DisplayName'],
            value_vars=['G0/G0', 'G1/G0', 'G2/G0'],
            var_name='Group',
            value_name='Value'
        )

        sns.barplot(
            ax=ax,
            data=melted,
            x='DisplayName',
            y='Value',
            hue='Group',
            palette=['red', 'blue', 'green']
        )

        ax.set_ylim(0, max_display)

        # Add labels only to bars exceeding max_display
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if height > max_display:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        max_display - 50,  # slightly below top
                        f'{height:.0f}%',
                        ha='center',
                        va='top',
                        fontsize=11,
                        color='white',
                        rotation=90
                    )

        ax.set_title(title, fontsize=16)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel('Normalized to Group 0 (%)', fontsize=12)

        if ax.get_legend() is not None:
            ax.get_legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.95),
        fontsize=14
    )

    fig.suptitle(
        "Mean Feature Values by Group (Normalized to Group 0)",
        fontsize=20,
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(f"{output_dir}/Normalized_Subplots.png", dpi=300)
    plt.close(fig)


plot_normalized_subplots(merged_df, output_dir="figures/k3_summary")

# ================================
# Heatmaps (Z-score version)
# ================================
zscore_df = features_df.apply(zscore, axis=1)

plt.figure(figsize=(16, 8))
sns.heatmap(zscore_df.T, annot=False, cmap="vlag", center=0, cbar_kws={'label': 'Z-score'})
plt.title("Z-Score Heatmap of All Features by Cluster")
plt.tight_layout()
plt.savefig(f"{output_dir}/Heatmap_All_Features_ZScore.png", dpi=300)
plt.close()

# Morphological z-score heatmap
morph_z = zscore_df.loc[zscore_df.index.isin(morph_features)]
if not morph_z.empty:
    plt.figure(figsize=(14, 6))
    sns.heatmap(morph_z.T, annot=False, cmap="vlag", center=0, cbar_kws={'label': 'Z-score'})
    plt.title("Z-Score Heatmap of Morphological Features by Cluster")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Heatmap_Morphological_Features_ZScore.png", dpi=300)
    plt.close()

# Kinetic z-score heatmap
kinetic_z = zscore_df.loc[zscore_df.index.isin(kinetic_features)]
if not kinetic_z.empty:
    plt.figure(figsize=(14, 6))
    sns.heatmap(kinetic_z.T, annot=False, cmap="vlag", center=0, cbar_kws={'label': 'Z-score'})
    plt.title("Z-Score Heatmap of Kinetic Features by Cluster")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Heatmap_Kinetic_Features_ZScore.png", dpi=300)
    plt.close()
