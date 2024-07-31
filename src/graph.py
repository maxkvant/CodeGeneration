from typing import List, Dict

class Step:
    def __init__(self,
                 step_id: str,
                 description: str,
                 dependencies: List[str],
                 input_vars: List[str],
                 output_vars: List[str],
                 additional_info):
        self.step_id = step_id
        self.desiption = description
        self.dependencies = dependencies
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.additional_info = additional_info

    def __str__(self):
      return f"Step({repr(self.__dict__)})"

    def __repr__(self):
      return str(self)