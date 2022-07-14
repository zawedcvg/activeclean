import pandas as pd

class Cleaner:
    def __init__(self, process_cleaned_df, own_filepath):
        self.process_cleaned_df = process_cleaned_df #user defined function
        self.cleaned_data_filepath = own_filepath + "SampleToClean_Iteration.xlsx"

    '''remove dirty samples if the probability of dirty is more than a threshold'''
    def update_with_cleaned_data(self, X_full, Y_full, dirty_sample_indices):
        # Process cleaned sample from user
        unprocessed_cleaned_sample_df = pd.read_excel(self.cleaned_data_filepath, index_col=0)
        unprocessed_cleaned_sample_df.index.name = 'Index'
        cleaned_sample_df = self.process_cleaned_df(unprocessed_cleaned_sample_df)

        # Update existing data with cleaned data
        full_data_df = pd.DataFrame(X_full, columns=["Movie", "Plot", "Genres"])
        full_data_df["Y label"] = Y_full
        full_data_df.loc[dirty_sample_indices, :] = cleaned_sample_df[:]
        Y_full = full_data_df["Y label"].values.tolist()
        X_data_df = full_data_df.drop(columns=["Y label"])
        X_full = list(X_data_df.itertuples(index=False))
        X_full = [tuple(namedtuple) for namedtuple in X_full]
        return (X_full, Y_full)

    def provide_sample(self,X_full, Y_full, dirty_sample_indices):
        full_data = pd.DataFrame(X_full, columns=["Movie", "Plot", "Genres"])
        full_data["Y label"] = Y_full
        dirty_sample_df = full_data.filter(items=dirty_sample_indices, axis=0)
        dirty_sample_df.to_excel(self.cleaned_data_filepath)
        print("Please clean the data in " + self.cleaned_data_filepath)
