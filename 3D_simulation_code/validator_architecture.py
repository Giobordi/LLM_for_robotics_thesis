from enum import Enum
import pickle
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
import yaml
from time import sleep
from tqdm import tqdm
from langchain.chains.llm import LLMChain
from langchain.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings, ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
import pandas as pd 
import re
from langchain_core.messages import BaseMessage
class ModelVersion(Enum) :
    GPT_35 = "gpt-3.5-turbo-0125"
    GPT_4O = "gpt-4o"
    CLAUDE_35_SONNET = "claude-3-5-sonnet-20240620"


class ModelLLM(Enum) : 
    AZURE = "azure"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

PROMPT_PATH = os.path.join(os.path.dirname(__file__),"..", "prompts", "3D_simulation")
MODEL = ModelVersion.CLAUDE_35_SONNET
API_TYPE = ModelLLM.ANTHROPIC


class FlowState(BaseModel):
    input : str 
    corrections : str = ""
    rate_limit : int = 0
    plan : str = ""
    valid : bool = False
    behavior_tree : str = ""

def get_room_sequence(beh_tree : str) -> str :
    return "\n".join(re.findall(r"GoToPose.*?location='(.*?)'", beh_tree))

  
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

def get_chain(system_path : str, output_model : BaseModel= None) : 
    prompt = get_system_with_user_input(system_path)
    llm = get_model(model=MODEL, api_type=API_TYPE).with_structured_output(schema=output_model, method="json_mode")
    return prompt | llm

def create_planning_graph(environment : str) -> CompiledGraph: 
    """
    Create a workflow graph for the task expansion and plan creation
    """
    workflow = StateGraph(FlowState)

    def nl_plan(state: FlowState) : 
        class NaturalLanguagePlan(BaseModel):
            movement_plan: str  
        chain = get_chain(os.path.join(PROMPT_PATH, environment, "nl_plan.yml"), output_model=NaturalLanguagePlan)      
        plan : NaturalLanguagePlan  = chain.invoke({"input" : state.input})
        return {"plan" : plan.movement_plan }
        
    
    def plan_validation(state: FlowState) : 
        class PlanValidation(BaseModel):
            valid : bool
            reasoning : str 
        chain = get_chain(system_path = os.path.join(PROMPT_PATH, environment, "plan_validator.yml"), output_model=PlanValidation)   
        validation : PlanValidation = chain.invoke({"input" : state.input , "plan" : state.plan})
        return {"valid" : validation.valid  , "corrections" : validation.reasoning }
        
    
    def plan_correction(state: FlowState) : 
        class PlanCorrection(BaseModel):
            corrected_plan : str
        chain = get_chain(os.path.join(PROMPT_PATH, environment, "plan_correction.yml"), output_model=PlanCorrection)
        corrected_plan : PlanCorrection= chain.invoke({"input" : state.input , "plan" : state.plan, "evaluation" : state.corrections})
        return {"plan" : corrected_plan.corrected_plan, "rate_limit" : state.rate_limit + 1}
    
    def code_conversion(state: FlowState) : 
        class CodeConversion(BaseModel):
            actions : list
        chain = get_chain(os.path.join(PROMPT_PATH, environment, "code_conversion.yml"), output_model=CodeConversion)
        conversion : CodeConversion= chain.invoke({"input" : state.input , "plan" : state.plan})
        
        beh_tree = """<root main_tree_to_execute = "MainTree" >
        <BehaviorTree ID="MainTree">
        <Sequence name="sequence">\n""" + "\n".join(conversion.actions).replace(">","/>") + """\n</Sequence>
        </BehaviorTree>
        </root>"""
        
        return {"input" : state.input , "behavior_tree" : beh_tree, "valid" : state.valid , "plan" : state.plan}



    def is_valid(state: FlowState)-> Literal['code_conversion', 'plan_correction']:
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




if __name__ == "__main__":
    
    environments = ["house"] 
    actions = [6]
    
    for environment in environments:
        for action in actions:
            
            app = create_planning_graph(environment)
            init_time = time.time()
            result = []
            try : 
                file_path = os.path.join(os.path.dirname(__file__), "..", "3D_dataset", f"dataset_{environment}_{action}actions.pkl")
                with open(file_path, 'rb') as file:
                    trajectory_commands = list(pickle.load(file)[0])                
                    if len(trajectory_commands) == 20 : 
                        trajectory_commands = trajectory_commands * 3
            except Exception as e: 
                continue
            trajectory_commands = ["i'm on the bathtub, go to my room and take my phone from the desk and bring it to me, then bring to me a bottle of water from the fridge"]
            for traj in tqdm(trajectory_commands, 
                             desc=f"Processing {environment}_{action}actions",
                            colour="green"):
                input = {"input" : traj}
                for chunk in app.stream(input):
                    res = chunk 
                res = chunk['code_conversion']
                tmp = {
                    "input" : traj,
                    "nl_plan" : res['plan'],
                    "valid" : res['valid'],
                    "code_conversion" : res['behavior_tree'],
                    "rooms_sequence" : get_room_sequence(res['behavior_tree'])
                    
                }
                result.append(tmp)
            save_directory = os.path.join(os.path.dirname(__file__), "..", "results", "3D_simulation", "validator_architecture", f"{environment}")
            end_time = time.time()
            seconds =  (end_time - init_time)  / len(trajectory_commands)
            ## create a directory if it does not exist
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            pd.DataFrame(result       
                        ).to_excel(os.path.join(save_directory, f"{MODEL.value}_{environment}_{action}actions_mean_{seconds}_sec.xlsx"), index=False)
    