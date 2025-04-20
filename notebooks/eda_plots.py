import matplotlib.pyplot as plt
import seaborn as sns

def sns_styles():
    sns.set(style='whitegrid', palette='muted', font_scale=1.1)
    plt.rcParams['figure.figsize']=(10,6)

def graph_pipe_line(df):
    sns_styles()

    # Default payment Balance
    plt.figure()
    sns.countplot(x='default payment next month', data=df)
    plt.title('Default Payment - Class Balance')
    plt.xlabel('Default')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['No Default', 'Default'])
    plt.show()

    # Age Distribution Hist Plot
    plt.figure()
    sns.histplot(x=df['AGE'], kde=True, bins=30, color='skyblue')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.show()

    # Limit Balance Distribution 
    plt.figure()
    sns.boxplot(x=df['LIMIT_BAL'])
    plt.title('Credit Limit Distribution')
    plt.xlabel('Limit Bal')
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(16,10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f', center=0)
    plt.title('Correlation Matrix')
    plt.show()

    # Means by Default k de plot
    mean_features=['BILL_MEAN', 'PAY_AMT_MEAN']
    for item in mean_features:
        plt.figure()
        sns.kdeplot(data=df, x=item, hue='default payment next month', common_norm=False, fill=True)
        plt.title(f'{item} Distribution by Default Status')
        plt.xlabel(item)
        plt.show()
