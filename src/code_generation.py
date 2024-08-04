from typing import List, Dict
from graph import Step

class CodeGenerator:
  def generate_source(self, step, step_dependencies, parameters, additonal_lines):
    pass

class MainCodeGenerator(CodeGenerator):
    def generate_source(self, step: Step, step_dependencies, parameters, additonal_lines=[]):
        # TODO: implement
        pass

class ValidationCodeGenerator(CodeGenerator):
  def generate_source(self, step: Step, step_dependencies: List[Step], parameters, additonal_lines=[]):
      input_params = step.input_vars
      output_params = step.output_vars

      validation_code = "import pandas as pd\n"
      for dep in step_dependencies:
          validation_code += f"from step_{dep.step_id} import step_{dep.step_id}\n"
      validation_code += f"from step_{step.step_id} import step_{step.step_id}\n\n"

      validation_code += "\n"
      for name, value in parameters.items():
          validation_code += f'{name} = {value}\n'
      validation_code += "\n"

      validation_code += "def validate_step():\n"

      for dep in step_dependencies:
          dep_inputs = ", ".join(dep.input_vars)
          dep_outputs = ", ".join(dep.output_vars)
          validation_code += f"    {dep_outputs} = step_{dep.step_id}({dep_inputs})\n"

      input_values = ", ".join(input_params)
      output_values = ", ".join(output_params)
      validation_code += f"    {output_values} = step_{step.step_id}({input_values})\n"
      validation_code += f"    print({output_values})\n"

      validation_code += "\nif __name__ == '__main__':\n"
      validation_code += "    validate_step()\n"
      return validation_code