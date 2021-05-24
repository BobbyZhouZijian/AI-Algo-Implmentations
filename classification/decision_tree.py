import math
import numpy as np
from util import calculate_entropy

class DecisionTree:

    '''Implement Decision Tree using the C45 algorithm'''
    def __init__(self):

    def preprocess_continuous(self, df, column_name, entropy):
        if df[column_name].unique() <= 20:
            unique_values = sorted(df[column_name].unique())
        else:
            unique_values = []

            df_mean = df[column_name].mean()
            df_std = df[column_name].std()
            df_min = df[column_name].min()
            df_max = df[column_name].max()

            scales = list(range(-3, 4, 1))
            for scale in scales:
                if df_mean + scale * df_std > df_min and df_mean + scale * df_std < df_max:
                    unique_values.append(df_mean + scale * df_std)
            unique_values.sort()
        
        subset_gainratios = []

        if len(unique_values) == 1:
            winner_threshold = unique_values[0]
            df[column_name] = np.where(df[column_name] <= winner_threshold, "<=" + str(winner_threshold), ">" + str(winner_threshold))
            return df
        
        for i in range(0, len(unique_values)-1):
            threshold = unique_values[1]

            subset1 = df[df[column_name] <= threshold]
            subset2 = df[df[column_name] > threshold]

            subset1_rows = subset1.shape[0]
            subset2_rows = subset2.shape[0]

            total_instances = df.shape[0]

            subset1_probability = subset1_rows / total_instances
            subset2_probability = subset2_rows / total_instances

            threshold_gain = entropy - subset1_probability * calculate_entropy(subset1) - subset2_probability * calculate_entropy(subset2)

            threshold_splitinfo = - subset1_probability * math.log(subset1_probability, 2) - subset2_probability * math.log(subset2_probability, 2)
            gainratio = threshold_gain / threshold_splitinfo
            subset_gainratios.append(gainratio)

            winner_one = subset_gainratios.index(max(subset_gainratios))

            winner_threshold = unique_values[winner_one]
            df[column_name] = np.where(df[column_name] <= winner_threshold, "<=" + str(winner_threshold), ">" + str(winner_threshold))

            return df


    def find_decision(self, df):
        resp_obj = self.find_gains(df)
        gains = list(resp_obj['gains'].values())
        entropy = resp_obj['entropy']

        winner_index = gains.index(max(gains))

        winner_name = df.columns[winner_index]
        return winner_name, df.shape[0], entropy
    
    def find_gains(self, df):
        decision_classes = df['Decision'].unique()

        entropy = calculate_entropy(df)

        num_columns = df.shape[1]
        num_instances = df.shape[0]

        gains = []

        for i in range(num_columns-1):
            column_name = df.columns[i]
            column_type = df[column_name].dtype

            if column_type != 'object':
                df = self.preprocess_continuous(df, column_name, entropy)
            classes = df[column_name].value_counts()

            splitinfo = 0
            gain = entropy

            for j in range(len(classes)):
                current_class = classes.keys().tolist()[j]
                subdataset = df[df[column_name] == current_class]
                num_subset_instances = subdataset.shape[0]
                class_probability = num_subset_instances / num_instances

                subset_entropy = calculate_entropy(subdataset)
                gain = gain - class_probability * subset_entropy

                splitinfo = splitinfo - class_probability * math.log(class_probability, 2)

            if splitinfo == 0:
                splitinfo == 100 # an arbitrarily large value
            gain = gain / splitinfo
            gains.append(gain)        
        
        resp_obj = {}
        resp_obj['gains'] = {}

        for idx, feature in enumerate(df.columns[0:-1]):
            resp_obj['gains'][feature] = gains[idx]
        
        resp_obj['entropy'] = entropy

        return resp_obj

    def create_branch(self, current_class, subdataset, )