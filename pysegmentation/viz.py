import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from dython.nominal import associations
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from typing import Union

def plot_qq_plot(
    x: np.ndarray, 
    figsize: tuple[int, int]=(8, 6), 
    title: str="Q-Q Plot (with seaborn)",
    xlabel: str="Theoretical Quantiles",
    ylabel: str="Sample Quantiles",
    legend: bool=True) -> plt.Figure:
    """
    Plot a Q-Q (Quantile-Quantile) plot to compare the distribution of a sample with a normal distribution.

    Parameters
    ----------
    x : np.ndarray
        The input data array to be compared against the normal distribution.
    figsize : tuple[int, int], optional
        The size of the matplotlib figure (default is (8, 6)).
    title : str, optional
        The title of the plot (default is "Q-Q Plot (with seaborn)").

    Returns
    -------
    fig : plt.Figure
        The matplotlib Figure object containing the Q-Q plot.

    Notes
    -----
    - The function uses seaborn for plotting.
    - The blue dots represent the sample quantiles versus the theoretical quantiles.
    - The red dashed line is the theoretical line for a perfect normal distribution.
    """
    (osm, osr), (slope, intercept, r) = stats.probplot(x, dist="norm") # osm: theoretical, osr: ordered sample

    # Calculate the theoretical line
    line = slope * np.array(osm) + intercept

    # Plot using seaborn
    sns.set_theme(rc={'axes.facecolor': '#fcf0dc'}, style='darkgrid')
    fig = plt.figure(figsize=figsize)

    # Scatter plot of the sample data
    sns.scatterplot(x=osm, y=osr, s=60, color='dodgerblue')

    # Plot the theoretical line
    sns.lineplot(x=osm, y=line, color="red", linestyle="--", label="Theoretical Line")

    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.title(title, fontsize=18)
    plt.grid(True)
    plt.legend()
    return fig



def corr_matrix(data: pd.DataFrame) -> dict:
    a = associations(
    data,
    nominal_columns='auto',
    plot=False,
    title="Correlation Matrix (Mixed Types)",
    figsize=(20, 20)
)
    return a

def plot_corr_matrix(
    data: pd.DataFrame, 
    figsize: tuple[int, int]=(12, 10), 
    title: str='Correlation Matrix',
    legend: bool=True) -> tuple[np.ndarray, plt.Figure]:
    """
    Plot a correlation matrix heatmap for a given DataFrame, supporting mixed data types.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data for which the correlation matrix will be computed.
    figsize : tuple[int, int], optional
        The size of the matplotlib figure (default is (12, 10)).
    title : str, optional
        The title of the plot (default is 'Correlation Matrix').

    Returns
    -------
    tuple[np.ndarray, plt.Figure]
        - The computed correlation matrix (numpy array or DataFrame).
        - The matplotlib Figure object containing the heatmap.

    Notes
    -----
    - The function uses a custom colormap for visualization.
    - Only the lower triangle of the correlation matrix is displayed for clarity.
    - The function supports mixed data types using the `associations` function from the `dython` package.
    """
    a = corr_matrix(data)
    # Reset background style
    sns.set_style('whitegrid')

    # Calculate the correlation matrix excluding the 'CustomerID' column
    corr = a['corr']

    # Define a custom colormap
    colors = ['#ff6200', '#ffcaa8', 'white', '#ffcaa8', '#ff6200']
    my_cmap = LinearSegmentedColormap.from_list('custom_map', colors, N=256)

    # Create a mask to only show the lower triangle of the matrix (since it's mirrored around its 
    # top-left to bottom-right diagonal)
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask, k=1)] = True

    # Plot the heatmap
    fig = plt.figure(figsize=figsize)
    sns.heatmap(corr, mask=mask, cmap=my_cmap, annot=True, center=0, fmt='.2f', linewidths=2)
    plt.title(title, fontsize=14)
    return a['corr'], fig


def plot_count_null(
    data: pd.DataFrame, 
    figsize: tuple[int, int] = (12, 10), 
    color: str = '#ff6200', 
    title: str = 'Count Null',
    legend: bool=True) -> tuple[pd.DataFrame, plt.Figure]:
    """
    Plot a bar chart showing the distribution of the number of null values per row in a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame to analyze for null values.
    figsize : tuple[int, int], optional
        The size of the matplotlib figure (default is (12, 10)).
    color : str, optional
        The color of the bars in the plot (default is '#ff6200').
    title : str, optional
        The title of the plot (default is 'Count Null').

    Returns
    -------
    tuple[pd.DataFrame, plt.Figure]
        - A DataFrame with the number of nulls per row and their counts.
        - The matplotlib Figure object containing the bar plot.
    """
    # Count the number of null values in each row
    num_null_col = data.isnull().sum(axis=1)
    # Count the frequency of each unique number of nulls
    c = num_null_col.value_counts().reset_index()
    c.columns = ['num_null', 'count']
    # Sort by the number of nulls in ascending order for better visualization
    c.sort_values(by='num_null', ascending=True, inplace=True)
    # Set seaborn style for the plot
    sns.set_style('whitegrid')
    # Create the figure
    fig = plt.figure(figsize=figsize)
    # Plot the bar chart
    sns.barplot(y='num_null', x='count', data=c, color=color)
    # Set the plot title
    plt.title(title, fontsize=14)
    return c, fig


def plot_categorical_feature(
    value: Union[pd.Series, np.ndarray, list], 
    figsize: tuple[int, int] = (7, 3), 
    color: str = '#ff6200', 
    title: str = 'Categorical Feature') -> plt.Figure:
    """
    Plot a bar chart for a categorical feature, showing both count and percentage for each category.

    Parameters
    ----------
    value : Union[pd.Series, np.ndarray, list]
        The categorical data to plot. Can be a pandas Series, numpy array, or list.
    figsize : tuple[int, int], optional
        Size of the matplotlib figure (default is (7, 3)).
    color : str, optional
        Color of the bars in the plot (default is '#ff6200').
    title : str, optional
        Title of the plot (default is 'Categorical Feature').

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the bar plot.
    """
    # Set seaborn theme for consistent plot style
    sns.set_theme(rc={'axes.facecolor': '#fcf0dc'}, style='darkgrid')
    
    # Convert input to pandas Series if it is not already
    if not isinstance(value, pd.Series):
        value = pd.Series(value)
    
    # Count occurrences of each category (including NaN)
    c1 = value.value_counts(dropna=False).reset_index()
    c1.columns = ['value', 'count']
    
    # Calculate percentage for each category (including NaN)
    c2 = value.value_counts(dropna=False, normalize=True).reset_index()
    c2.columns = ['value', 'percentage']
    
    # Merge count and percentage into a single DataFrame
    c = c1.merge(c2, on='value', how='left')
    c.columns = ['value', 'count', 'percentage']
    
    # Replace NaN values in 'value' column with string 'NaN' for display
    c['value'] = c['value'].fillna('NaN')
    
    # Create the figure
    fig = plt.figure(figsize=figsize)
    
    # Plot the bar chart
    sns.barplot(y='value', x='count', data=c, color=color)
    plt.ylabel('')
    
    # Annotate each bar with count and percentage
    for i, (v1, v2) in enumerate(zip(c['count'], c['percentage'])):
        plt.text(v1, i, f"{v1} ({v2:.2%})", va='center')
    
    # Set the plot title
    plt.title(title, fontsize=14)
    return fig

def plot_categorical_features(
    data: pd.DataFrame, 
    cat_cols: list[str], 
    figsize: tuple[int, int]=(10, 5), 
    color: str='#ff6200') -> list[plt.Figure]:
    """
    Plot a bar chart for each categorical feature in a DataFrame.
    """
    figs = []
    for col in cat_cols:
        value = data[col].values 
        fig = plot_categorical_feature(value, figsize=figsize, color=color, title=col)
        figs.append(fig)
    return figs

def plot_numerical_feature(
    value: Union[pd.Series, np.ndarray, list], 
    figsize: tuple[int, int]=(10,5), 
    color: str='#ff6200', 
    title: str='Numerical Feature',
    xlabel: str='Density',
    ylabel: str='Value') -> plt.Figure:
    """
    Plot a density and ECDF plot for a numerical feature.
    """
    sns.set_theme(rc={'axes.facecolor': '#fcf0dc'}, style='darkgrid')

    fig, ax = plt.subplots(figsize=figsize, ncols=2)
    _ = sns.kdeplot(value, ax=ax[0], color=color)
    ax[0].set_title('Density Plot')
    _ = sns.ecdfplot(value, ax=ax[1], color=color)
    ax[1].set_title('ECDF Plot')
    plt.title(title, fontsize=14)
    return fig

def plot_numerical_features(data: pd.DataFrame, num_cols: list[str], figsize: tuple[int, int]=(10, 5), color: str='#ff6200') -> list[plt.Figure]:
    figs = []
    for col in num_cols:
        value = data[col].values 
        fig = plot_numerical_feature(value, figsize=figsize, color=color, title=col)
        figs.append(fig)
    return figs


def plot_distribution_cluster(
    data: pd.DataFrame, col: str, 
    bins: list[float], bin_labels: list[str], 
    cluster_column: str='cluster_kmean', 
    colors: list[str]=None, 
    cluster_labels=None, 
    cut_point: float=None, 
    cut_label: str=None, 
    y_min: float=None, 
    y_max: float=None,
    title: str='Distribution Cluster',
    xlabel: str='Value',
    ylabel: str='Percentage (%)',
    legend: bool=True) -> plt.Figure:

    """
    Plot the distribution of a numerical feature across different clusters.

    Parameters:
    -----------
    data : pd.DataFrame
        Input DataFrame containing the feature and cluster assignments.
    col : str
        Name of the numerical feature column to plot.
    bins : list[float]
        List of bin edges for discretizing the feature.
    bin_labels : list[str]
        List of labels for the bins.
    cluster_column : str, default='cluster_kmean'
        Name of the column containing cluster labels.
    colors : list[str], optional
        List of colors for each cluster.
    cluster_labels : list[str], optional
        List of custom labels for each cluster.
    cut_point : float, optional
        Value at which to draw a vertical cut line.
    cut_label : str, optional
        Label for the cut line.
    y_min : float, optional
        Minimum y-value for the cut line.
    y_max : float, optional
        Maximum y-value for the cut line.
    title : str, default='Distribution Cluster'
        Title of the plot.
    xlabel : str, default='Value'
        Label for the x-axis.
    ylabel : str, default='Percentage (%)'
        Label for the y-axis.
    legend : bool, default=True
        Whether to display the legend.

    Returns:
    --------
    plt.Figure
        The matplotlib Figure object containing the plot.
    """
    sns.set_theme(rc={'axes.facecolor': '#fcf0dc'}, style='darkgrid')
    fig = plt.figure(figsize=(15, 6))
    clusters = sorted(data[cluster_column].unique())
    for cluster in clusters:
        cluster_data = data[data[cluster_column] == cluster][col]
        label = cluster_labels[cluster] if cluster_labels else f'Nhóm {cluster}'
        binned = pd.cut(cluster_data, bins=bins, labels=bin_labels, include_lowest=True, right=False)
        percent_dist = binned.value_counts(normalize=True).sort_index() * 100
        color = colors[cluster] if colors else None
        sns.barplot(percent_dist, color=color, label=label, alpha=0.3)
    if cut_point is not None:
        plt.vlines(x=cut_point, ymin=y_min, ymax=y_max, color='black', linestyles='--', label=cut_label)
    plt.ylabel('Percentage (%)')
    plt.legend(loc='upper right')
    plt.title(f'Distribution of {col} in {len(clusters)} Clusters')
    return fig

def plot_kde_cluster(data: pd.DataFrame, col: str, 
                     cluster_column: str='cluster_kmean', 
                     colors: list[str]=None,
                     cluster_labels: list[str]=None, 
                     cut_point: float=None, 
                     cut_label: str=None, 
                     y_min: float=None, 
                     y_max: float=None) -> plt.Figure:
    """
    Plot Kernel Density Estimation (KDE) curves for different clusters.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame containing the data.
    col : str
        Name of the column to plot.
    cluster_column : str
        Name of the column containing cluster labels.
    colors : list, optional
        List of colors for each cluster.
    cluster_labels : list, optional
        List of labels for each cluster.
    cut_point : float, optional
        Value at which to draw a vertical cut line.
    cut_label : str, optional
        Label for the cut line.
    y_min : float, optional
        Minimum y-value for the cut line.
    y_max : float, optional
        Maximum y-value for the cut line.
    """
    plt.figure(figsize=(10, 6))
    
    # Lấy các nhóm unique
    clusters = sorted(data[cluster_column].unique())
    
    # Vẽ KDE cho từng nhóm
    for i, cluster in enumerate(clusters):
        cluster_data = data[data[cluster_column] == cluster][col]
        label = cluster_labels[i] if cluster_labels else f'Nhóm {cluster}'
        color = colors[i] if colors else None
        sns.kdeplot(cluster_data, color=color, label=label, linewidth=2)
    
    # Vẽ đường cắt nếu có
    if cut_point is not None:
        plt.vlines(x=cut_point, ymin=y_min, ymax=y_max, color='black', linestyles='--', 
                  label=cut_label if cut_label else f'Cut point = {cut_point}')
    
    plt.title(f"Density Plot for {len(clusters)} Clusters \n with {col}")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()
# Tính phần trăm



def find_threshold(
    data: pd.DataFrame, 
    col: str, 
    cluster_column: str='cluster_kmean', 
    right_clusters: list[int]=[3]) -> float:
    
    """
    Find the optimal cut-off point between two groups based on the intersection of their KDEs.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data.
    col : str
        Name of the column to find the cut-off point for.
    cluster_column : str
        Name of the column containing cluster labels.
    right_clusters : list[int], optional
        List of cluster labels considered as the "right" group (default is [3]).

    Returns
    -------
    float
        The optimal cut-off point between the two groups.
    """
    data_class_right = data[data[cluster_column].isin(right_clusters)][col].values
    data_class_left = data[~data[cluster_column].isin(right_clusters)][col].values

    # KDE
    kde_right = gaussian_kde(data_class_right)
    kde_left = gaussian_kde(data_class_left)

    # Trục x
    x_vals = np.linspace(min(data[col]), max(data[col]), 1000)
    
    # Mật độ
    pdf_right = kde_right(x_vals)
    pdf_left = kde_left(x_vals)
    
    def find_intersection(x, y1, y2):
        idx = np.argmin(np.abs(y1 - y2))
        return x[idx]

    threshold = find_intersection(x_vals, pdf_right, pdf_left)
    return threshold