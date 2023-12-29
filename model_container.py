import pickle
import os
import multiprocessing
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import date_formater as dtf

class ModelContainer:
    def __init__(
        self,
        model=None,
        grid_search=False,
        random_state=32,
        param_grid = { # my standard RF GridSearch
                'bootstrap': [False,True],
                'n_estimators': [600,1000, 1400,2000],
                'max_depth': [None,40,80,120],
                'min_samples_split': [4,8,10,12],
                'min_samples_leaf': [2,3,4,5,6],
                'max_features': [2,4,6,'auto','log2', 'sqrt']
            }
    ):
        self.model = model
        if model is not None:
            self.model.random_state = random_state
        self.features = []

        self.personas_removed = False
        self.low_correlation_removed = False
        self.grid_search = grid_search

        if self.grid_search:
            self.n_jobs = multiprocessing.cpu_count()-1
            self.param_grid = param_grid
        
    ## PUBLIC METHODS
    def fit(self, X_train, y_train,folds=3):
        if self.grid_search:
            self.model = GridSearchCV(estimator=self.model, param_grid=self.param_grid, cv=folds, n_jobs=self.n_jobs, verbose=2)
        self.model.fit(X_train,y_train)
        if self.grid_search:
            self.model = self.model.best_estimator_

    def predict(self,X_test):
        return pd.DataFrame(self.model.predict(X_test),columns=['Pred'])


    def save_model(self, model, features):
        filepath = model + '.pkl'
        with open(filepath, 'wb') as file:
            pickle.dump(self.model, file)
        pd.DataFrame(self.features,columns=['Features']).to_csv(features)

    def load_model(self, model, features):
        with open(model, 'rb') as file:
            self.model = pickle.load(file)
        self.features = pd.read_csv(features)['Features'].values

    def evaluate(self, X_test, y_test):
        # Calculation of key metrics
        predictions = self.predict(X_test)
        r2 = r2_score(y_test, predictions)

        # Metrics Dictionary
        metrics = {
            'Hyper_Params': self.model.get_params(),
            'MAE': mean_absolute_error(y_test, predictions),
            'MAPE': self._mean_absolute_percentage_error(y_test, predictions),
            'MSE': mean_squared_error(y_test, predictions),
            'R2': r2,
            'adj(R2)': self._adjusted_r_squared(r2,X_test.shape[0],X_test.shape[1]),
            'Train_Acc': self.model.score(X_train, y_train),
            'Test_Acc': self.model.score(X_test, y_test)
        }

        # Report Writing
        report_callback = lambda counter_file_path, report_path: \
            self._write_report_file(counter_file_path,report_path,metrics)

        metrics['WriteStatus'] = str(self._generate_report(report_callback))
        # Return dataframe with metrics
        return pd.DataFrame([metrics])

    # EVAL METHODS
    def _mean_absolute_percentage_error(self,y_true, y_pred): 
        # Avoid division by zero
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        non_zero_mask = y_true != 0
        return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

    def _adjusted_r_squared(self,r_squared, n_samples, n_features):
        return 1 - (1 - r_squared) * (n_samples - 1) / (n_samples - n_features - 1)

    # REPORT WRITING
    def _write_report_file(self,counter_file_path,report_path,metrics):
        with open(report_path, 'w') as file:
            file.write(f'Model Type: {type(self.model)}\n')
            file.write(f'Datasets Used: {self.dataset_info}\n')
            file.write(f'Platform column added to dataset\n')
            file.write(f'personas_removed: {self.personas_removed}\n')
            file.write(f'low_correlation_removed {self.low_correlation_removed}\n\n')
            file.write(f'Grid search used: {self.grid_search}\n\n')
            file.write('Model Hyperparameters:\n')
            for param, value in metrics['Hyper_Params'].items():
                file.write(f'{param}: {value}\n')
            file.write('\n')
            file.write(f'Mean Absolute Error: {metrics["MAE"]}\n')
            file.write(f'Mean Squared Error: {metrics["MSE"]}\n')
            file.write(f'R-squared: {metrics["R2"]}\n')
            file.write(f'Adj(R-squared): {metrics["adj(R2)"]}\n')
            file.write('\n')
            file.write(f'Train Accuracy: {metrics["Train_Acc"]}\n')
            file.write(f'Test Accuracy: {metrics["Test_Acc"]}\n')

    def _write_experiment_count(self,counter_file_path):
        exp_num = self._read_exp_counter(counter_file_path) + 1
        with open(counter_file_path, 'w') as f:
            f.write(str(exp_num))

    def _read_exp_counter(self,counter_file_path):
        if os.path.exists(counter_file_path):
            with open(counter_file_path, 'r') as f:
                return int(f.read().strip())
        return 0

    def _set_report_path(self,counter_file_path):
        exp_num = self._read_exp_counter(counter_file_path) + 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f'exp/exp{exp_num}_{timestamp}.txt'

    def _remove_report_file(self,report_path):
        if os.path.exists(report_path):
            os.remove(report_path)
            print(f"File {report_path} removed due to an error.")

    def _generate_report(self,report_callback):
        counter_file_path = 'exp/experiments.txt'
        try:
            os.makedirs('exp', exist_ok=True)
            report_path = self._set_report_path(counter_file_path)
            report_callback(counter_file_path,report_path)
            self._write_experiment_count(counter_file_path)
            return "Success"
        except Exception as e:
            print(f"An error occurred while creating the report: {e}")
            self._remove_report_file(report_path)
            return "Abort"


    def _combine_dfs(self,df1,df2):
        df1[df1.attrs['name']] = 1
        df1[df2.attrs['name']] = 0
        df2[df1.attrs['name']] = 0
        df2[df2.attrs['name']] = 1
        all_columns = sorted(set(df1.columns).union(set(df2.columns)))
        all_columns = list(set(all_columns))

        # Reindex both DataFrames to have the same columns, filling missing values with 0
        df1_aligned = df1.reindex(columns=all_columns, fill_value=0)
        df2_aligned = df2.reindex(columns=all_columns, fill_value=0)

        # Concatenate the aligned DataFrames
        dataframe = pd.concat([df1_aligned, df2_aligned], ignore_index=True)
        return dataframe

    def get_data_splits(self,df1,df2=None,remove_personas=True,remove_low_corr=True,random_state = 42):
        if df2 is None:
            self.dataset_info = '['+df1.attrs['name']+']'
            dataframe = pd.DataFrame(df1)
        else:
            self.dataset_info = '['+df1.attrs['name']+', '+df2.attrs['name']+']'
            # Combine the columns from both DataFrames
            dataframe = self._combine_dfs(df1,df2)
    
        if remove_personas:
            self.personas_removed = True
            dataframe = pd.DataFrame(dataframe.drop(columns=['Annemieke','Marco','Nelleke','Petra','Thijmen']))

        if remove_low_corr:
            self.low_correlation_removed = True
            corr_matrix = dataframe.corr()
            high_corr_with_attendance = corr_matrix['AttendRatio'].abs() > 0.11
            columns2use = [x for x,y in zip(high_corr_with_attendance.index,high_corr_with_attendance.values) if y == True]
            dataframe = pd.DataFrame(dataframe[columns2use])
    
        y = dataframe['AttendRatio']
        X = dataframe.drop('AttendRatio', axis=1)
        self.features = X.columns
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def generate_simulation(self,artiest,genre,zaal,posttype,platform,event_date,progress):
        datapoints = []
        date_list = dtf.date_simulation(event_date)
        for i,publish_date in enumerate(date_list):
            progress(int(i/len(date_list) * 100))
            df = pd.DataFrame(0, index=np.arange(1), columns=self.features)
            df[artiest] = 1
            df[genre] = 1
            df[zaal] = 1
            df[platform] = 1
            df[posttype] = 1
            days_b4_event, month, week_day, hour = dtf.create_date_inputs(str(publish_date),event_date)
            df['days_b4_event'] = days_b4_event
            df['month'] = month
            df['week_day'] = week_day
            df['hour'] = hour
            datapoints.append(df)
        return pd.DataFrame(date_list,columns=['Dates']), pd.concat(datapoints, ignore_index=True)

    def generate_empty(self):
        datapoints = []
        df = pd.DataFrame(0, index=np.arange(1), columns=self.features)
        df['Dates'] = 0
        return df

   



        
