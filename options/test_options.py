from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        self.parser.add_argument('--eval_path', type=str, required=True, help='path to the eval data')
        BaseOptions.initialize(self)
        self.isTrain = False
