import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from imdb import IMDB
from activecleanprocessor import ActiveCleanProcessor
import ast

from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

plot_filepath, genre_filepath, yahoo_filepath = "../data/plot.list", "../data/imdb-genres.list", \
                                                "../data/ydata-ymovies-movie-content-descr-v1_0.txt"

print("Processing data...")
# Script
imdb_data_processor = IMDB(plot_filepath, genre_filepath, yahoo_filepath)
X, Y = imdb_data_processor.get_comedy_and_horror_movie_data()
full_data = (X, Y)
full_indices = list(range(len(Y)))
train_indices, test_indices = train_test_split(full_indices, test_size=0.20)
X_train, Y_train = [X[i ] for i in train_indices], [Y[i] for i in train_indices]
dirty_data = (X_train, Y_train)
dirty_indices = train_indices
clean_indices = []
indices = [dirty_indices, clean_indices, test_indices]
total_labels = []
print("Done processing")

# User defined functions
def data_to_features(data):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    text = [
        unicodedata.normalize("NFKD", i[0] + " " + i[1]).encode(
            "ascii", "ignore"
        )
        for i in data
    ]

    return vectorizer.fit_transform(text)

def process_cleaned_df(df):
    def convertStrToList(text):
        lst = ast.literal_eval(text)
        assert type(lst) == list
        return lst

    df["Genres"] = df["Genres"].apply(lambda x: convertStrToList(x))
    return df

# User Defined Variables
own_filepath = "C:\\Users\\isabe\\OneDrive\\Desktop\\ActiveCleanFiles\\"
num_records_to_clean = 20
batch_size = 5
step_size = 0.1  # learning_rate

model = SGDClassifier(
        loss="hinge", alpha=0.000001, max_iter=200, fit_intercept=True, warm_start=True)

print("Initialising ActiveClean...")
ActiveClean = ActiveCleanProcessor(model, full_data, indices, batch_size, own_filepath, step_size, data_to_features,
                                   process_cleaned_df, binary_crossentropy)

ActiveClean.start(dirty_data, num_records_to_clean)
print("Done initialising")

print("Starting retraining...")
ActiveClean.runNextIteration(num_records_to_clean)


