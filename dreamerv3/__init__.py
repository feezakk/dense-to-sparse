import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent))

# from .agent_teacher import Agent
from .agent_teacher import Agent as agent_teacher
from .agent_student import Agent as agent_student
from .agent_student_bisim import Agent as agent_student_bisim