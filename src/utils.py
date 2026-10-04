import os
import sys
import pickle

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
            e,
            sys
        )