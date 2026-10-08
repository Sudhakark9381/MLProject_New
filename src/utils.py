import os
import sys
import pickle

from sklearn.metrics import r2_score

from src.exception import CustomException


def save_object(file_path, obj):

    try:

        # Get directory path
        dir_path = os.path.dirname(file_path)

        # Create directory if it doesn't exist
        os.makedirs(
            dir_path,
            exist_ok=True
        )

        # Save object
        with open(
            file_path,
            "wb"
        ) as file_obj:

            pickle.dump(
                obj,
                file_obj
            )

    except Exception as e:

        raise CustomException(
            e,sys
        )

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            # print("Model name:", list(models.keys())[i])
            # print("Model object:", model)
            # print("Model type:", type(model))
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report


    except Exception as e:
        raise CustomException(e, sys)