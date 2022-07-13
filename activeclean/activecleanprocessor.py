import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score

from activeclean.detector import Detector
from activeclean.sampler import DetectorSampler, UniformSampler
from activeclean.cleaner import Cleaner
from activeclean.updater import Updater


class ActiveCleanProcessor:
    def __init__(self, user_clf, full_data, indices, batch_size, ownFilepath, step_size, featuriser, process_cleaned_df,
                 calculate_loss):
        """
        Initialises an ActiveClean object
        :param user_clf: A classifier that the user hopes to improve
        :param full_data: A dataframe
        :param indices: A tuple of lists (dirty_indices, clean_indices)
        :param batch_size: Size of data to sample
        :param own_filepath: User defined filepath to store xlsx files
        """
        self.clf = user_clf

        self.dirty_indices = indices[0]
        self.clean_indices = indices[1]
        self.test_indices = indices[2]

        # training data to clean
        self.X_full = full_data[0]
        self.Y_full = full_data[1]

        self.batch_size = batch_size
        self.step_size = step_size
        self.ownFilepath = ownFilepath

        self.featuriser = featuriser
        self.process_cleaned_df = process_cleaned_df
        self.calculate_loss = calculate_loss

        self.detector = Detector(SGDClassifier(loss="log", alpha=1e-6, max_iter=200, fit_intercept=True))
        self.detectorsampler = DetectorSampler(featuriser(self.X_full), self.dirty_indices, self.detector, batch_size)
        self.uniformsampler = UniformSampler(full_data, self.dirty_indices, batch_size)
        self.cleaner = Cleaner(process_cleaned_df, ownFilepath)

        self.total_labels = []
        self.dirty_sample_indices = []
        self.sampling_prob = []

    def start(self, dirty_data, num_records_to_clean):
        if num_records_to_clean < len(dirty_data):
            return Exception("Please provide smaller batch size")

        x_dirty, y_dirty = dirty_data
        featurised_x_dirty = self.featuriser(x_dirty)

        # Train user clf first
        self.clf = self.clf.fit(featurised_x_dirty, y_dirty)

        # Update user of accuracy
        self.printReport()

        # Sample dirty data
        self.dirty_sample_indices, self.sampling_prob = self.uniformsampler.sample()

        # Provide sample to clean to user in Excel file
        self.cleaner.provide_sample(self.X_full, self.Y_full, self.dirty_sample_indices)

    def runNextIteration(self, num_records_to_clean):
        user_want_to_keep_cleaning = 1
        while len(self.clean_indices) < num_records_to_clean and user_want_to_keep_cleaning:
            # Update data after cleaning
            self.X_full, self.Y_full = self.cleaner.update_with_cleaned_data(self.X_full, self.Y_full,
                                                                             self.dirty_sample_indices)

            # Update data
            self.dirty_indices = [index for index in self.dirty_indices if index not in self.dirty_sample_indices]
            for index in self.dirty_sample_indices:
                self.clean_indices.append(index)

            # updater is work in progress
            updater = Updater(self.clf, self.batch_size, self.step_size, self.calculate_loss)
            self.clf = updater.retrain(self.featuriser(self.X_full), self.Y_full)

            # Update user of clf report
            self.printReport()

            # Update detector sampler
            self.total_labels.extend([(index, True) for index in self.dirty_sample_indices])
            self.detector.update_classifier(self.total_labels, self.featuriser(self.X_full))
            self.detectorsampler.updateDetector(self.detector)

            # Sample dirty data
            self.dirty_sample_indices, self.sampling_prob = self.detectorsampler.sample()

            # Provide dirty samples to user to clean
            self.cleaner.provide_sample(self.X_full, self.Y_full, self.dirty_sample_indices)

            user_want_to_keep_cleaning = input("Continue? Return 1 if Yes or 0 if No\n")

        print("Done")

    def printReport(self):
        print(("Number Cleaned So Far ", len(self.clean_indices)))
        x_test = self.featuriser([self.X_full[i] for i in self.test_indices])
        y_test = [self.Y_full[i] for i in self.test_indices]
        ypred = self.clf.predict(x_test)
        print(("Prediction Freqs ", np.sum(ypred), np.shape(ypred)))
        print((classification_report(y_test, ypred)))
        print(("Accuracy ", accuracy_score(y_test, ypred)))

    def getClassifier(self):
        return self.clf
