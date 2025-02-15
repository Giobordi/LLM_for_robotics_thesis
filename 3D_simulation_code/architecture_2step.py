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

PROMPT_PATH = os.path.join(os.path.dirname(__file__),"..", "prompts", "3D_simulation")
MODEL = ModelVersion.GPT_4O
API_TYPE = ModelLLM.OPENAI


class FlowState(BaseModel):
    input : str 
    plan : str = ""
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


def create_planning_graph() -> CompiledGraph: 
    """
    Create a workflow graph for the task expansion and plan creation
    """
    workflow = StateGraph(FlowState)

    def nl_plan(state: FlowState) : 
        class NaturalLanguagePlan(BaseModel):
            movement_plan: str  
        chain = get_chain(os.path.join(PROMPT_PATH, environment, "nl_plan.yml"), output_model=NaturalLanguagePlan)      
        plan = chain.invoke({"input" : state.input})
        return {"plan" : plan.movement_plan}
   
    def code_conversion(state: FlowState) : 
        class CodeConversion(BaseModel):
            actions : list
        chain = get_chain(os.path.join(PROMPT_PATH, environment, "code_conversion.yml"), output_model=CodeConversion)
        conversion = chain.invoke({"input" : state.input , "plan" : state.plan})
        
        beh_tree = """<root main_tree_to_execute = "MainTree" >
        <BehaviorTree ID="MainTree">
        <Sequence name="sequence">""" + "\n".join(conversion.actions) + """</Sequence>
        </BehaviorTree>
        </root>"""
        
        return {"input" : state.input , "behavior_tree" : beh_tree, "plan" : state.plan}


        
    ##create the nodes 
    workflow.add_node("nl_plan", nl_plan)
    workflow.add_node("code_conversion", code_conversion)

    ##create the edges

    workflow.set_entry_point("nl_plan")
    workflow.add_edge("nl_plan","code_conversion")
    workflow.add_edge("code_conversion", END)

    plan_graph = workflow.compile()
    return plan_graph

if __name__ == "__main__":

    environments = [ "restaurant" ]
    actions = [2,4,6]
    
    for environment in environments:
        for action in actions:         
            app = create_planning_graph()
            init_time = time.time()
            result = []
            file_path = os.path.join(os.path.dirname(__file__), "..", "3D_dataset", f"dataset_{environment}_{action}actions.pkl")
            with open(file_path, 'rb') as file:
                trajectory_commands = list(pickle.load(file)[0])
                if len(trajectory_commands) == 20 : 
                    trajectory_commands = trajectory_commands * 3
            for traj in tqdm(trajectory_commands, 
                             desc=f"Processing {environment}_{action}actions",
                            colour="green"):
                input = {"input" : traj}
                for chunk in app.stream(input):
                    res = chunk 
                res = chunk['code_conversion']
                result.append(
                    {
                    "input" : traj,
                    "nl_plan" : res['plan'],
                    "code_conversion" : res['behavior_tree'],
                    "rooms_sequence" : get_room_sequence(res['behavior_tree'])      
                    }
                )
                #sleep(3)
            save_directory = os.path.join(os.path.dirname(__file__), "..", "results", "3D_simulation", "architecture_2step", f"{environment}")
            end_time = time.time()
            seconds =  (end_time - init_time) / len(trajectory_commands)
            ## create a directory if it does not exist
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            pd.DataFrame(result       
                        ).to_excel(os.path.join(save_directory, f"{MODEL.value}_{environment}_{action}actions_mean_{seconds}_sec.xlsx"), index=False)