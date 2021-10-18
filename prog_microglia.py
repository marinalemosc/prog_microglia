import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# importing microglia morphology data
microglia_df = pd.read_csv("C:/Users/marin/OneDrive/Documentos/Prog/dados_microglia.csv", decimal=',')
# removing analysis index  column  
del microglia_df['Índice de Analise']
# modifying columns names
microglia_df.columns = ['Group','Animal','Segments','End-points','Lengh','N Cells','Density','Span Ratio','Circularity','Area']
microglia_df['Area'] = microglia_df['Area'].astype(float)
microglia_df.head()

# separating skeleton and fraclac parameters 
skeleton_df = microglia_df[['Group','Segments','End-points','Lengh']]
fraclac_df = microglia_df[['Group','Density','Span Ratio','Circularity','Area']]
print(fraclac_df.head(2),2*'\n', skeleton_df.head(2))

# plot1 -> n cells x groups
sns.set_theme(style = 'whitegrid')
sns.boxplot(x=microglia_df['Group'],y=microglia_df['N Cells'],palette='Set2', width=0.3,linewidth=0.5)
sns.stripplot(x=microglia_df['Group'], y=microglia_df['N Cells'],color='black', jitter=0.2, size=2)

# plot n2 -> skeleton parameters x group 
fig, axs = plt.subplots(ncols=3,figsize=(20,5))
sns.boxplot(data=skeleton_df, y='Segments', x='Group', ax=axs[0],palette='Set2', width=0.3,linewidth=0.5)
sns.boxplot(data=skeleton_df, y='End-points', x='Group',ax=axs[1],palette='Set2', width=0.3,linewidth=0.5)
sns.boxplot(data=skeleton_df, y='Lengh', x='Group', ax=axs[2],palette='Set2', width=0.3,linewidth=0.5)

# plot n3 -> fraclac parameters x group 
fig, axs = plt.subplots(ncols=4, figsize=(25,5))
sns.boxplot(data=fraclac_df, y='Density', x='Group', ax=axs[0],palette='Set2', width=0.3,linewidth=0.5)
sns.boxplot(data=fraclac_df, y='Span Ratio', x='Group',ax=axs[1],palette='Set2', width=0.3,linewidth=0.5)
sns.boxplot(data=fraclac_df, y='Circularity', x='Group', ax=axs[2],palette='Set2', width=0.3,linewidth=0.5)
sns.boxplot(data=fraclac_df, y='Area', x='Group', ax=axs[3],palette='Set2', width=0.3,linewidth=0.5)

# Plot n -> paramters x parameters 
g=sns.PairGrid(microglia_df,diag_sharey= False, hue='Group').add_legend()
g.map_lower(sns.kdeplot)
g.map_upper(sns.regplot)
g.map_diag(sns.kdeplot,lw=1,fill=True)

# creating correlation df and ploting it 
corr_df = microglia_df.drop(['Group','Animal',], axis=1).corr()
sns.heatmap(corr_df, xticklabels=corr_df.columns, yticklabels=corr_df.columns, cmap='Spectral_r')

# From the heatmap, creating a plot correlating the 2 parapeters with the strongest correlation 
plot = sns.jointplot(data=skeleton_df ,x="Segments", y="End-points",height=5, hue='Group')
plot.ax_marg_x.set_xlim(0, 700)
plot.ax_marg_y.set_ylim(0, 700)

# starting modeling 
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# testing modeling 
train, test = train_test_split(microglia_df, test_size = 0.33, stratify = microglia_df['Group'], random_state = 42)
# sns.pairplot(train, hue='Group', height = 2, palette = 'colorblind')
# creating models variables 
pd.Categorical(microglia_df.Group)
X_train = train[['Segments','End-points','Lengh','Density','Circularity','Area']]
y_train = train.Group
X_test = test[['Segments','End-points','Lengh','Density','Circularity','Area']]
y_test = test.Group

# creating the decision Tree 
dectree = DecisionTreeClassifier(max_depth = 4, random_state = 1)
dectree.fit(X_train,y_train)
pred=dectree.predict(X_test)
print('DecTree Acc',"{:.3f}".format(metrics.accuracy_score(pred,y_test)))
# Decision tree accuracy is 0.35, which is kind of low and not ideal.

# plot n7 -> ilustrating the decision tree
features = ['Segments','End-points','Lengh','Density','Circularity','Area']
classes = ['Jovem', 'Idoso', 'Idoso + GH']
plt.figure(figsize = (10,15))
plot_tree(dectree, feature_names=features, class_names=classes, filled = True)
plt.show()

# add plot n8-> confusion matrix, expliciting decision errors 
disp = metrics.plot_confusion_matrix(dectree, X_test, y_test, display_labels=classes, cmap=plt.cm.Blues, normalize=None)
disp.ax_.set_title('DecTree Confusion Matrix');
# the decision tree displays lots of errors, expecially in the group 'Idoso'

"""In summary, based upon the given parameters there is no accuracy to predict the experimental group which each cell belong. Therefore, in order to further develop this modeling, a new analysis must be done with new parameters  that describe a stronger correlation."""