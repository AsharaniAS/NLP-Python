# Model evaluation
import json
import os

class ModelEvaluator:
    def __init__(self,trainer):
        self.trainer = trainer

    def evaluate_model(self):
        eval_results = self.trainer.evaluate()
        self.save_results(eval_results)
        return eval_results
    
    def save_results(self, eval_results):
        results_path = os.path.join(self.output_dir, 'eval_results.json')   
        with open('eval_results', 'w') as f:
            json.dump(eval_results, f, indent=4) 