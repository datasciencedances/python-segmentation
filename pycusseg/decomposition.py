from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def reduce_dimension(data:pd.DataFrame, n_components=2, top_k=3):
    pca = PCA(n_components=n_components)

    # Fitting and transforming the original data to the new PCA dataframe
    data_pca = pca.fit_transform(data)

    # Creating a new dataframe from the PCA dataframe, with columns labeled PC1, PC2, etc.
    data_pca = pd.DataFrame(data_pca, columns=['PC'+str(i+1) for i in range(pca.n_components_)])

    # Adding the CustomerID index back to the new PCA dataframe
    data_pca.index = data.index

    def highlight_topk(column):
        topk = column.abs().nlargest(top_k).index
        return ['background-color:  green' if i in topk else '' for i in column.index]

    # Create the PCA component DataFrame and apply the highlighting function
    pc_df = pd.DataFrame(pca.components_.T, columns=['PC{}'.format(i+1) for i in range(pca.n_components_)],  
                        index=data.columns)

    pc_df = pc_df.style.apply(highlight_topk, axis=0)
    return pca, data_pca, pc_df

def plot_pca_variance(df_scaled: pd.DataFrame, threshold: float = 0.9, point_step=5, figsize=(20, 15)):
    pca = PCA().fit(df_scaled)

    # Calculate the Cumulative Sum of the Explained Variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    threshold = threshold
    optimal_k = np.argmax(cumulative_explained_variance >= threshold) + 1

    # Set seaborn plot style
    sns.set(rc={'axes.facecolor': '#fcf0dc'}, style='darkgrid')

        # Plot the cumulative explained variance against the number of components
    fig = plt.figure(figsize=figsize)

    # Bar chart for the explained variance of each component
    barplot = sns.barplot(x=list(range(1, len(cumulative_explained_variance) + 1)),
                        y=explained_variance_ratio,
                        color='#fcc36d',
                        alpha=0.8)

    # Line plot for the cumulative explained variance
    lineplot, = plt.plot(range(0, len(cumulative_explained_variance)), cumulative_explained_variance,
                        marker='o', linestyle='--', color='#ff6200', linewidth=2)

    # Plot optimal k value line
    optimal_k_line = plt.axvline(optimal_k - 1, color='red', linestyle='--', label=f'Optimal k value = {optimal_k}') 

    # Set labels and title
    plt.xlabel('Number of Components', fontsize=14)
    plt.ylabel('Explained Variance', fontsize=14)
    plt.title('Cumulative Variance vs. Number of Components', fontsize=18)

    # Customize ticks and legend
    plt.xticks(range(0, len(cumulative_explained_variance)), rotation=60)
    plt.legend(handles=[barplot.patches[0], lineplot, optimal_k_line],
            labels=['Explained Variance of Each Component', 'Cumulative Explained Variance', f'Optimal k value = {optimal_k}'],
            loc=(0.62, 0.1),
            frameon=True,
            framealpha=1.0,  
            edgecolor='#ff6200')  

    # Display the variance values for both graphs on the plots
    x_offset = -0.3
    y_offset = 0.01
    for i, (ev_ratio, cum_ev_ratio) in enumerate(zip(explained_variance_ratio, cumulative_explained_variance)):
        if i % point_step == 0:  # 👈 chỉ hiện nhãn mỗi 5 cột
            plt.text(i, ev_ratio, f"{ev_ratio:.2f}", ha="center", va="bottom", fontsize=10)
            if i > 0:
                plt.text(i + x_offset, cum_ev_ratio + y_offset, f"{cum_ev_ratio:.2f}", ha="center", va="bottom", fontsize=10)

    plt.grid(axis='both')   
    return optimal_k, fig