from enum import Enum
from langchain_core.prompts import (  ## prompt and messages templates
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    )

from langchain_core.output_parsers import (  ## output parsers
    JsonOutputParser,
    ListOutputParser,
    PydanticOutputParser,
    StrOutputParser , # parse top likely string.
    )
import os 
import yaml
from dotenv import load_dotenv
load_dotenv("LLM_for_robotics_thesis\\.env.local")
import operator
import functools
import uuid
from typing import Annotated, Literal, TypedDict
from langgraph.graph import END, StateGraph, MessageGraph 
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import add_messages
#from IPython.display import Image, display
import time
from time import sleep
from tqdm import tqdm
from langchain.chains.llm import LLMChain
from langchain.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings, ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic

from tqdm import tqdm
import pandas as pd 
import pickle
import re


class ModelVersion(Enum) :
    GPT_35 = "gpt-3.5-turbo-0125"
    GPT_4O = "gpt-4o"
    CLAUDE_35_SONNET = "claude-3-5-sonnet-20240620"

class ModelLLM(Enum) : 
    AZURE = "azure"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    
    
PROMPT_PATH = os.path.join(os.path.dirname(__file__),"..", "prompts", "real_environment")
MODEL = ModelVersion.GPT_4O
API_TYPE = ModelLLM.OPENAI

class Utils : 
    @staticmethod
    def get_room_sequence(beh_tree : str) -> str :
        return "\n".join(re.findall(r"GoToPose.*?location='(.*?)'", beh_tree))

    @staticmethod    
    def get_model(model : str, api_type : ModelLLM = ModelLLM.OPENAI): 
        if api_type == ModelLLM.AZURE:   
            azure_llm = AzureChatOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    base_url=os.getenv("AZURE_OPENAI_BASE_URL"),
                    model=model.value,
                    temperature=0.0, 
                    )
            return azure_llm

        elif api_type == ModelLLM.OPENAI: 
            openai_llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            #verbose=True,
            model=model.value,
            temperature=0.0, 
            timeout=50,
            )
            return openai_llm

        elif api_type == ModelLLM.ANTHROPIC:
            anthropic = ChatAnthropic(model=model.value,
                        temperature=0,
                        max_tokens=1024,
                        api_key=os.getenv("ANTHROPIC_API_KEY"),
                        
                        )
            return anthropic
            
    @staticmethod
    def get_system_with_user_input(path : str) -> ChatPromptTemplate :
        with open(path, 'r') as file:
        # Load the YAML contents
            prompt = yaml.safe_load(file)

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(prompt["system"], template_format="mustache"),
                HumanMessagePromptTemplate.from_template(prompt["user"]),
            ])
        return prompt
    @staticmethod
    def get_chain(system_path : str, output_model : BaseModel= None) : 
        prompt = Utils.get_system_with_user_input(system_path)
        llm = Utils.get_model(model=MODEL, api_type=API_TYPE).with_structured_output(schema=output_model, method="json_mode")
        return prompt | llm



class TwoNodeArchitecture : 
    name = "architecture_2step"
    class FlowState(BaseModel):
        task : str 
        plan : str = ""
        behavior_tree : str = ""
        environment : str = ""
    
    def __init__(self):
        self.process_graph = self.create_planning_graph()
    
    def create_planning_graph(self) -> CompiledGraph: 
        """
        Create a workflow graph for the task expansion and plan creation
        """
        workflow = StateGraph(TwoNodeArchitecture.FlowState)

        def nl_plan(state: TwoNodeArchitecture.FlowState) : 
            class NaturalLanguagePlan(BaseModel):
                movement_plan: str  
            chain = Utils.get_chain(os.path.join(PROMPT_PATH, state.environment, "nl_plan.yml"), output_model=NaturalLanguagePlan)      
            plan = chain.invoke({"input" : state.task})
            return {"plan" : plan.movement_plan}
    
        def code_conversion(state: TwoNodeArchitecture.FlowState) : 
            class CodeConversion(BaseModel):
                actions : list
            chain = Utils.get_chain(os.path.join(PROMPT_PATH, state.environment, "code_conversion.yml"), output_model=CodeConversion)
            conversion = chain.invoke({"input" : state.task , "plan" : state.plan})
            
            beh_tree = """<root main_tree_to_execute = "MainTree" >
            <BehaviorTree ID="MainTree">
            <Sequence name="sequence">""" + "\n".join(conversion.actions) + """</Sequence>
            </BehaviorTree>
            </root>"""
            
            return {"input" : state.task , "behavior_tree" : beh_tree, "plan" : state.plan}


            
        ##create the nodes 
        workflow.add_node("nl_plan", nl_plan)
        workflow.add_node("code_conversion", code_conversion)

        ##create the edges

        workflow.set_entry_point("nl_plan")
        workflow.add_edge("nl_plan","code_conversion")
        workflow.add_edge("code_conversion", END)

        plan_graph = workflow.compile()
        return plan_graph
    
    def run(self, input_str : str, environment : str) : 
        input_obj = {
            "task" : input_str,
            "environment" : environment
        }
        for chunk in self.process_graph.stream(input_obj):
            res = chunk 
        res = chunk['code_conversion']
        return  {
            "task" : input_str,
            "nl_plan" : res['plan'],
            "code_conversion" : res['behavior_tree'],
            "rooms_sequence" : Utils.get_room_sequence(res['behavior_tree'])      
            }
            
class ValidatorArchitecture: 
    name = "validator_architecture"
    class FlowState(BaseModel):
        input : str 
        plan : str = ""
        corrections : str = ""
        rate_limit : int = 0
        valid : bool = False
        environment : str = ""
        behavior_tree : str = ""
    
    def __init__(self):
        self.process_graph = self.create_planning_graph()
    
    def create_planning_graph(self) -> CompiledGraph:
        """
        Create a workflow graph for the task expansion and plan creation
        """
        workflow = StateGraph(ValidatorArchitecture.FlowState)

        def nl_plan(state: ValidatorArchitecture.FlowState) : 
            class NaturalLanguagePlan(BaseModel):
                movement_plan: str  
            chain = Utils.get_chain(os.path.join(PROMPT_PATH, state.environment, "nl_plan.yml"), output_model=NaturalLanguagePlan)      
            plan : NaturalLanguagePlan  = chain.invoke({"input" : state.input})
            return {"plan" : plan.movement_plan }
            
        
        def plan_validation(state: ValidatorArchitecture.FlowState) : 
            class PlanValidation(BaseModel):
                valid : bool
                reasoning : str 
            chain = Utils.get_chain(system_path = os.path.join(PROMPT_PATH, state.environment, "plan_validator.yml"), output_model=PlanValidation)   
            validation : PlanValidation = chain.invoke({"input" : state.input , "plan" : state.plan})
            return {"valid" : validation.valid  , "corrections" : validation.reasoning }
            
        
        def plan_correction(state: ValidatorArchitecture.FlowState) : 
            class PlanCorrection(BaseModel):
                corrected_plan : str
            chain = Utils.get_chain(os.path.join(PROMPT_PATH, state.environment, "plan_correction.yml"), output_model=PlanCorrection)
            corrected_plan : PlanCorrection= chain.invoke({"input" : state.input , "plan" : state.plan, "evaluation" : state.corrections})
            return {"plan" : corrected_plan.corrected_plan, "rate_limit" : state.rate_limit + 1}
        
        def code_conversion(state: ValidatorArchitecture.FlowState) : 
            class CodeConversion(BaseModel):
                actions : list
            chain = Utils.get_chain(os.path.join(PROMPT_PATH, state.environment, "code_conversion.yml"), output_model=CodeConversion)
            conversion : CodeConversion= chain.invoke({"input" : state.input , "plan" : state.plan})
            
            beh_tree = """<root main_tree_to_execute = "MainTree" >
            <BehaviorTree ID="MainTree">
            <Sequence name="sequence">\n""" + "\n".join(conversion.actions) + """\n</Sequence>
            </BehaviorTree>
            </root>"""
            
            return {"input" : state.input , "behavior_tree" : beh_tree, "valid" : state.valid , "plan" : state.plan}

        
        def is_valid(state: ValidatorArchitecture.FlowState)-> Literal['code_conversion', 'plan_correction']:
            if state.valid:
                return "code_conversion"
            elif state.rate_limit > 2 :
                return "code_conversion"
            else : 
                return "plan_correction"
    
        ##create the nodes 
        workflow.add_node("nl_plan", nl_plan)
        workflow.add_node("plan_validation", plan_validation)
        workflow.add_node("plan_correction", plan_correction)
        workflow.add_node("code_conversion", code_conversion)

        ##create the edges

        workflow.set_entry_point("nl_plan")
        
        workflow.add_edge("nl_plan","plan_validation")
        workflow.add_conditional_edges("plan_validation", is_valid)
        workflow.add_edge("plan_correction","plan_validation")    
        
        workflow.add_edge("code_conversion", END)

        plan_graph = workflow.compile()
        #display(Image(plan_graph.get_graph().draw_mermaid_png()))
        return plan_graph
    
    def run(self, input_str : str, environment : str) : 
        input_obj = {
            "input" : input_str,
            "environment" : environment
        }
        for chunk in self.process_graph.stream(input_obj):
            res = chunk
        res = chunk['code_conversion']
        return  {
            "task" : input_str,
            "nl_plan" : res['plan'],
            "code_conversion" : res['behavior_tree'],
            "rooms_sequence" : Utils.get_room_sequence(res['behavior_tree'])      
            }
        
        

if __name__ == "__main__":
    environments = ["lab"]
    actions = [2]
    
    ##read a pickle file
    with open(os.path.join(os.path.dirname(__file__), "..", "3D_dataset", "dataset_house_4actions.pkl"), 'rb') as file:
        trajectory_commands = list(pickle.load(file)[0])
        print(trajectory_commands)
    # for environment in environments:
    #         for action in actions:         
    #             for arch in [ ValidatorArchitecture()]:
    #                 result = []
    #                 try : 
    #                     file_path = os.path.join(os.path.dirname(__file__), "..", "real_environment_dataset",environment, f"dataset_{environment}_{action}actions.pkl")
    #                     with open(file_path, 'rb') as file:
    #                         trajectory_commands = list(pickle.load(file)[0])                
    #                         if len(trajectory_commands) == 20 : 
    #                             trajectory_commands = trajectory_commands * 3
    #                 except Exception as e: 
    #                     continue
                    
    #                 for task in tqdm(trajectory_commands, 
    #                          desc=f"Processing {environment}_{action}actions",
    #                         colour="green"):
    #                     input = {"task" : task , "environment" : environment}
    #                     tmp =  arch.run(task, environment)
    #                     result.append(tmp)
    #                 save_directory = os.path.join(os.path.dirname(__file__), "..", "results", "real_environment", arch.name, f"{environment}")
    #                 ## create a directory if it does not exist
    #                 if not os.path.exists(save_directory):
    #                     os.makedirs(save_directory)
    #                 pd.DataFrame(result       
    #                             ).to_excel(os.path.join(save_directory, f"{MODEL.value}_{environment}_{action}actions.xlsx"), index=False)

        
