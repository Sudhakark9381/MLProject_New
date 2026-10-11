import os
import sys
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

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

# def evaluate_models(X_train, y_train, X_test, y_test, models, params):
#     try:
#         report = {}

#         for i in range(len(list(models))):
#             model = list(models.values())[i]

#             para = params(list(model.keys()))[i]

#             gs =GridSearchCV(model, para, cv=3)
#             gs.fit(X_train,y_train)

#             model.set_params(**gs.best_params_)


#             # print("Model name:", list(models.keys())[i])
#             # print("Model object:", model)
#             # print("Model type:", type(model))
#             model.fit(X_train, y_train)

#             y_train_pred = model.predict(X_train)

#             y_test_pred = model.predict(X_test)

#             train_model_score = r2_score(y_train, y_train_pred)

#             test_model_score = r2_score(y_test, y_test_pred)

#             report[list(models.keys())[i]] = test_model_score

#         return report


#     except Exception as e:
#         raise CustomException(e, sys)


def evaluate_models(
    X_train, y_train,
    X_test, y_test,
    models, params
):
    try:
        report = {}

        # Iterate through model names and model objects
        for model_name, model in models.items():

            # Get the hyperparameter grid for this model
            para = params[model_name]

            # Perform hyperparameter tuning when parameters exist
            if para:
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=para,
                    cv=3,
                    scoring="r2",
                    n_jobs=-1
                )

                gs.fit(X_train, y_train)

                # Apply the best hyperparameters
                model.set_params(**gs.best_params_)

                print(f"Model: {model_name}")
                print(f"Best parameters: {gs.best_params_}")

            # Train the model
            model.fit(X_train, y_train)

            # Generate predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R2 scores
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            # Store test score
            report[model_name] = test_score

            print(f"Training R2: {train_score:.4f}")
            print(f"Testing R2: {test_score:.4f}")
            print("-" * 50)

        return report

    except Exception as e:
        raise CustomException(e, sys) from e
