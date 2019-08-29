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
        self.model_config = model_config
        self.eval_config = eval_config
        self.overwrite = overwrite
        self.stopwatch = Stopwatch()

    def execute_model_pipeline(self):
        self.initialize_and_get_jsons()
        self.stopwatch.add('initialized and loaded JSON file(s)')
        self.validate_jsons()
        self.stopwatch.add('loaded and validated')
        self.generate_or_fetch_cv_data()
        self.stopwatch.add('generated or fetched CV sets')

        if self.model_dict['actions']['do_train_and_score_cv']:
            self.train_and_score_cv()
            self.stopwatch.add('trained and scored')

            if self.model_dict['actions']['do_evaluate']:
                self.evaluate_model()
                self.stopwatch.add('evaluated model')

            else:
                pass
                # print('per model.json, skipping model evaluation...')
        else:
            pass
            # print('per model.json, skipping model training, scoring, and evaluation')

        self.write_timing_data()

    def initialize_and_get_jsons(self):
        configs = InitializeTrial(self.model_config, self.eval_config, self.overwrite)
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
            print(self.evaluation_dict.get('to_plot', {}))
            plot.plot_all(
                self.evaluation_dict.get('to_plot', {}),
                self.model_dict, self.cv_scores, self.model_objects
            )

    def write_timing_data(self):
        self.stopwatch.write('{}/{}/time-stats.csv'.format(
            self.model_dict['models_dir'],
            self.model_dict['model_id'])
        )
