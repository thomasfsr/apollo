## Installation 

The package manager choosen for this project is UV for its simplicity and speed.  
However, for standard installation there is also a requirements.txt file.  
  
First step is to clone this repo:  
```bash
git clone https://www.github.com/thomasfsr/apollo  
cd apollo  
```  
After that you need to install the dependencies:  
If you choose UV you can do:  
```bash
uv sync
```  
This command will create the virtual environment and install all dependancies for you.  
  
If you choose to use pip:  
```bash
python -m venv .venv  
pip install -r requirements.txt  
```  
This command will also create the virtual environment and install all dependancies for you.  
  
## Run project  
This project uses Taskipy to manage the python scripts.  
The commands must be in the following order:  
  
```bash
task etl
```  
This task will extract the data of the pickle file and flatten to a dataset format and load to csv file in the data directory.  
  
```bash
task cluster
```  
This task will use t-sne to reduce dimensionality
of the data into 2 components, maintaining the 
semantic information. The data will be saved in
tsne_df.csv file.  
This will be used in the report app.  
  
```bash
task train
```  
This task will train the model for each combination
of hyperparameters. The result will be loaded in a 
CSV file with the name results.csv.
  
```bash
task mlf
```  
This command runs the MLFlow server.  
To open the MLFlow dashboard open the browser with 
the URI: localhost:5000  
  
```bash
task plot  
```  
This final command will run the streamlit app with
the report of the project. This report has all the 
visualization plots for Exploratory Data Analysis and futher analysis of the performance of the models.  
To open the dashboard open the URI: localhost:8501
  
## Overview  
This project focuses on analyzing medical image data to predict syndromes using embeddings extracted from a pre-trained CNN model. The workflow includes exploratory data analysis (EDA), normality checks, dimensionality reduction with T-SNE, and classification using a K-Nearest Neighbors (KNN) model. The goal is to validate the correlation between image embeddings and syndrome categories while addressing challenges like high-dimensional data and long-tailed distributions.  
  
## Key Features  
- **Exploratory Data Analysis (EDA):** Visualize syndrome distributions and validate data quality.  
- **Normality Testing:** Shapiro-Wilk tests to check embedding normality.  
- **Dimensionality Reduction:** T-SNE for 2D visualization of embeddings.  
- **KNN Classification:** Hyperparameter optimization (distance metrics, k-values) for syndrome prediction.  
- **Performance Evaluation:** ROC curves, AUC scores, F1-Score and per-syndrome analysis.  
  

