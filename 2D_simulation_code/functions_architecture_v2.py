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
import yaml
from time import sleep
from tqdm import tqdm
from langchain.chains.llm import LLMChain
from langchain.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings, ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
import pandas as pd 
from langchain_core.messages import BaseMessage
from tqdm import tqdm


class ModelLLM(Enum) : 
    AZURE = "azure"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "functions")
MODEL = "gpt-4o"
API_TYPE = ModelLLM.OPENAI


class FlowState(BaseModel):
    input : str 
    corrections : str = ""
    rate_limit : int = 0
    plan : str = ""
    valid : bool = False
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
        
    
    def plan_validation(state: FlowState) : 
        class PlanValidation(BaseModel): 
            valid : bool
            reasoning : str
        if state.rate_limit > 2 : 
            return {"valid" : False , "plan" : state.plan}
        chain = get_chain(system_path = os.path.join(PROMPT_PATH, "plan_validation.yml"), output_model=PlanValidation)   
        validation = chain.invoke({"input" : state.input , "plan" : state.plan})
        return {"valid" : validation.valid , "corrections" : str(validation.reasoning)}
        
    
    def plan_correction(state: FlowState) : 
        class PlanCorrection(BaseModel):
            actions : list
        chain = get_chain(os.path.join(PROMPT_PATH, "plan_correction.yml"), output_model=PlanCorrection)
        corrected_plan = chain.invoke({"input" : state.input , "plan" : state.plan, "evaluation" : state.corrections})
        return {"plan" : str(corrected_plan.actions), "rate_limit" : state.rate_limit +1}
    
    def code_conversion(state: FlowState) : 
        class CodeConversion(BaseModel): 
            functions : list
        chain = get_chain(os.path.join(PROMPT_PATH, "code_conversion.yml"), output_model=CodeConversion)
        conversion = chain.invoke({"input" : state.input , "plan" : state.plan})
        return {"input" : state.input , "code_plan" : str(conversion.functions), "plan" : state.plan}

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
    "Create a square with an arch on one side",
    "Create 3/4 of a circle and go to point x = 6 y= 9" ] * 3
    result = []
    chunk = None
    for traj in tqdm(trajectory_commands):
        try : 
            input = {"input" : traj}
            for chunk in app.stream(input):
                res = chunk
            result.append(dict(res['code_conversion']))
        except Exception as e :
            print(e) 
            continue

    pd.DataFrame(result).to_excel(f"results\\functions\\validator_architecture\\{MODEL}_validator_architecture.xlsx", index=False)   
    