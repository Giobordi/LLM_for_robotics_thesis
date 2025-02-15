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
load_dotenv(".env.local")
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


class ModelLLM(Enum) : 
    AZURE = "azure"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "functions")
MODEL = "claude-3-5-sonnet-20240620"
API_TYPE = ModelLLM.ANTHROPIC


class FlowState(BaseModel):
    input : str 
    plan : str = ""
    code_plan : list = ""
    
def get_model(model : str, api_type : ModelLLM = ModelLLM.OPENAI): 
    if api_type == ModelLLM.AZURE:   
        azure_llm = AzureChatOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                base_url=os.getenv("AZURE_OPENAI_BASE_URL"),
                model=model,
                temperature=0.0, 
                )
        return azure_llm

    elif api_type == ModelLLM.OPENAI: 
        openai_llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        #verbose=True,
        model=model,
        temperature=0.0, 
        timeout=50,
        )
        return openai_llm

    elif api_type == ModelLLM.ANTHROPIC:
        anthropic = ChatAnthropic(model=model,
                    temperature=0,
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

def get_chain(system_path : str, output_model : BaseModel = None) : 
    prompt = get_system_with_user_input(system_path)
    llm = get_model(model=MODEL, api_type=API_TYPE).with_structured_output(schema=output_model, method="json_mode")
    return prompt | llm 


def create_planning_graph() -> CompiledGraph: 
    """
    Create a workflow graph for the task expansion and plan creation
    """
    workflow = StateGraph(FlowState)

    def nl_plan(state: FlowState) : 
        class nlPlan(BaseModel): 
            plan : list 
        chain = get_chain(os.path.join(PROMPT_PATH, "nl_plan.yml"), output_model=nlPlan)      
        plan = chain.invoke({"input" : state.input})
        return {"plan" : str(plan.plan)}
    
    def code_conversion(state: FlowState) : 
        class CodeConversion(BaseModel): 
            functions : list
        chain = get_chain(os.path.join(PROMPT_PATH, "code_conversion.yml"), output_model=CodeConversion)
        conversion = chain.invoke({"input" : state.input , "plan" : state.plan})
        return {"input" : state.input , "code_plan" : str(conversion.functions), "plan" : state.plan}


        
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
    app = create_planning_graph()
    trajectory_commands = [
    "create a rectangle of dimension X = 3 m and Y = 1 m",
    "crea un quadrato di lunghezza 2 metri",
    "create a circle respecting the limits of the fields",
    "reach x = 2.5 and y= 9.0",
    "go to point x=6 y=7 passing through x=2 and y=2",
    "create a square and than a circle",
    "create a rectangle of x = 5 and y = 3 and than go to position x=9 and y=9",
    "create a U trajectory",
    "Create a square with an arch as side",
    "Create 3/4 of a circle and go to point x = 6 y= 9" ] * 3
    result = []
    chunk = None
    for traj in tqdm(trajectory_commands):
        input = {"input" : traj}
        for chunk in app.stream(input):
            res = chunk
        result.append(dict(res['code_conversion']))
        
    pd.DataFrame(result, columns = result[0].keys()        
                 ).to_excel(f"results\\functions\\architecture_2step\\{MODEL}_natural_plan_into_commands.xlsx", index=False)