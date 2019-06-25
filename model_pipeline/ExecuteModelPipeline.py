from matplotlib import rcParams
import sys

from CVData import CVData
from EvaluateModel import EvaluationData, EvaluateAndPlot
from InitializeTrial import InitializeTrial
from json_validator import validate_model_json, validate_eval_json
from Stopwatch import Stopwatch
from TrainAndScoreModel import TrainAndScoreModel

# Import custom matplotlib configs
sys.path.append('config')
from mpl_style import *
rcParams['figure.dpi'] = 96
rcParams['figure.figsize'] = (12,8)

class ExecuteModelPipeline:
    def __init__(self, model_config, eval_config, overwrite='N'):
        stopwatch = Stopwatch()
        self.initialize_and_get_jsons(model_config, eval_config, overwrite)
        stopwatch.add('initialized and loaded JSON file(s)')
        self.validate_jsons()
        stopwatch.add('loaded and validated')
        self.generate_or_fetch_cv_data()
        stopwatch.add('generated or fetched CV sets')

        if self.model_dict['actions']['do_train_and_score_cv']:
            self.train_and_score_cv()
            stopwatch.add('trained and scored')

            if self.model_dict['actions']['do_evaluate']:
                self.evaluate_model()
                stopwatch.add('evaluated model')

            else:
                print('per model.json, skipping model evaluation...')
        else:
            print('per model.json, skipping model training, scoring, and evaluation')

        self.write_timing_data(stopwatch)

    def initialize_and_get_jsons(self, model_config, eval_config, overwrite):
        configs = InitializeTrial(model_config, eval_config, overwrite)
        self.model_dict = configs.model_dict
        self.evaluation_dict = configs.evaluation_dict

    def validate_jsons(self):
        # Load and Validate the JSON
        validate_model_json(self.model_dict)

        if type(self.evaluation_dict) is not type(None):
            validate_eval_json(self.model_dict, self.evaluation_dict)

    def generate_or_fetch_cv_data(self):
        # Generate CV Sets
        cv = CVData(self.model_dict)
        self.model_data = cv.generate_cv_data()
        self.is_classification = cv.is_classification

    def train_and_score_cv(self):
        # Train and Score
        model = TrainAndScoreModel(self.model_dict, self.is_classification)
        model.cv_train_and_score(self.model_data['training'],
                                 self.model_data['scoring_only'])
        self.cv_scores = model.cv_scores
        self.model_objects = model.model_objects

        if self.model_dict['actions']['do_score_holdout']:
            self.holdout_scores = model.score_holdout(model.model_objects['full'],
                                                        self.model_data['holdout'])

    def evaluate_model(self):
        if type(self.evaluation_dict) is not type(None):
            # Evaluate Model
            plot = EvaluateAndPlot(
                self.evaluation_dict, self.cv_scores, self.is_classification
            )
            plot.plot_all(self.model_dict, self.cv_scores, self.model_objects)

    def write_timing_data(self, stopwatch):
        stopwatch.write('{}/{}/time-stats.csv'.format(
            self.model_dict['models_dir'],
            self.model_dict['model_id'])
        )
