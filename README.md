# tf-flower-classifications
    Isaac Perks
    06-17-2023

# Description
A basic tensorflow classification model for predicting flower classifications. Based on tensorflow tutorial.

- Import flower dataset from google storage
    - Wrap in a pandas dataframe while renaming columns
    - Removed our goal index to set as our training y-value
- Set up an input function to determine batch size, shuffle our data and return a dataset
- Create feature columns based on index keys of our dataframes
