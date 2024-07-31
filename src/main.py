import pandas as pd
import numpy as np
import json
from sklearn.decomposition import PCA

from graph import Step
from language_modeling import OpenAiLlamaApi, LlamaModel, PromptGenerator
from code_generation import ValidationCodeGenerator, MainCodeGenerator
from orchestrator import Orchestrator
from utils import get_dataset_info

EXAMPLE_STEP_SCRIPT = """
import pandas as pd
import pywt
from sklearn.preprocessing import StandardScaler

def step_40(Segments_normalized, Dec_levels):
    Features = []
    for segment in Segments_normalized:
        coeffs = pywt.wavedec(segment, 'db4', level=Dec_levels)
        features = [coefficient.mean() for coefficient in coeffs]
        Features.append(features)
    return StandardScaler().fit_transform(Features)
"""

EXAMPLE_VALIDATION_SCRIPT = """
import pandas as pd
from step_10 import step_10
from step_20 import step_20
from step_30 import step_30
from step_40 import step_40

def validate_step():
    csv_path = '/path/to/your/csv/file.csv'
    raw_data = step_10(csv_path)
    Segments = step_20(raw_data, SizeSegment=512)
    Segments_normalized = step_30(Segments)
    Features = step_40(Segments_normalized, Dec_levels=5)
    print(Features)

if __name__ == '__main__':
    validate_step()
"""

steps = [
    Step(
        step_id="10",
        description="Import raw data from CSV and segment it",
        dependencies=[],
        input_vars=["csv_path", "SizeSegment"],
        output_vars=["Segments"],
        additional_info="Use pandas to read the CSV and create segments of size SizeSegment."
    ),
    Step(
        step_id="20",
        description="Normalize the segmented data using MinMaxScaler",
        dependencies=["10"],
        input_vars=["Segments"],
        output_vars=["Segments_normalized"],
        additional_info="Segments is a list of 1D numpy arrays. Each segment should be normalized independently."
    ),
    Step(
        step_id="30",
        description="Extract features using wavelet decomposition",
        dependencies=["20"],
        input_vars=["Segments_normalized", "Dec_levels"],
        output_vars=["Features"],
        additional_info="Use pywavelets (pywt) library with 'db3' wavelet and specified Dec_levels."
    ),
    Step(
        step_id="40",
        description="Apply PCA for dimension reduction",
        dependencies=["30"],
        input_vars=["Features", "NC_pca"],
        output_vars=["PCA_Features", "pca"],
        additional_info="Use sklearn's PCA. Return both the transformed features and the PCA object."
    ),
    Step(step_id="50",
        description="Train model, evaluate, and calculate metrics",
        dependencies=["40"],
        input_vars=["PCA_Features", "kernel", "nu", "gamma"],
        output_vars=["FittedClassifier", "Prec_learn", "Prec_test"],
        additional_info="""
        1. Create labels: np.ones for learning data.
        2. Split data into train and test sets (80% train, 20% test).
        3. Create and fit a One-Class SVM classifier using sklearn.
        4. Predict labels for training data.
        5. Calculate error rate for training data.
        6. Predict labels for test data (assume all test data as anomaly, i.e., -1).
        7. Calculate error rate for test data.
        8. Calculate precision as 1 - error_rate for both training and test.
        Return the fitted classifier and both precision values.
        """
    )
]


def main():
    csv_path = 'learning-file_2.csv'
    raw_data = pd.read_csv(csv_path)
    dataset_info = get_dataset_info(raw_data)
    # Assume raw_data is a pandas DataFrame with 'timestamp' and 'signal' columns
    signal_data = raw_data['signal'].values

    # Adjust based on data size
    SizeSegment = min(512, len(signal_data) // 100)
    gamma = 'scale'  # Let sklearn choose an appropriate scale
    nu = 0.1  # This might need domain knowledge to set appropriately
    kernel = "rbf"  # This is often a good default

    # PCA
    # We'll use the signal data for PCA parameter calculation
    pca = PCA().fit(signal_data.reshape(-1, 1))
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    NC_pca = np.argmax(cumulative_variance_ratio >= 0.95) + 1

    Dec_levels = int(np.log2(SizeSegment)) - 3  # Adjust based on segment size
    
    parameters = {
        'csv_path': f"'{csv_path}'",
        "SizeSegment": f"{SizeSegment}",
        "gamma": f"'{gamma}'",
        "nu": f"{nu}",
        "kernel" : f"'{kernel}'",
        "NC_pca": f"{NC_pca}",
        "Dec_levels": f"{Dec_levels}",
    }

    with open('env.json', 'r') as f:
        credentials_dict = json.load(f)

    API_URL = "https://openrouter.ai/api/v1"
    API_KEY = credentials_dict["OPENROUTER_API_KEY"]
    MODEL_TAG = "meta-llama/llama-3-70b-instruct"
    llama_api = OpenAiLlamaApi(API_URL, API_KEY, MODEL_TAG)
    model = LlamaModel(llama_api)
    prompt_generator = PromptGenerator(EXAMPLE_STEP_SCRIPT, dataset_info)
    validation_code_genrator = ValidationCodeGenerator()
    main_code_generator = MainCodeGenerator()

    orchestrator = Orchestrator(
        model,
        prompt_generator,
        validation_code_genrator,
        main_code_generator,
        'out'
    )
    orchestrator.run_steps(steps, parameters)


if __name__ == '__main__':
    main()
