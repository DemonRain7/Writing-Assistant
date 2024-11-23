import re

from langchain_openai import AzureChatOpenAI
import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

import wikipedia

wikipedia.set_lang("en")

# Setting environment variables for LangSmith and Azure OpenAI
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "llm_ps"

os.environ["AZURE_OPENAI_API_KEY"] = "296848302c0b48658c716878f62371a8"
os.environ[
    "AZURE_OPENAI_ENDPOINT"] = "https://hack-for-fun.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2023-03-15-preview"

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    api_version="2023-03-15-preview",
    temperature=0.7,
    max_tokens=4000,
    timeout=10,
    max_retries=2,
)


# Define the state structure
class State(TypedDict):
    input_validate: bool
    output_validate: bool
    user_info: dict
    ps_draft: str
    domain_data: dict
    judge_result: str
    is_judged: bool

# Define the consultant step
def consultant_step(state: dict) -> dict:
    print("Consultant_Step:")
    basic_info_template = ("I am {template_name}, My GPA is {template_gpa}, My major is {template_major}. "
                           "I am in {template_university}, my target university is {template_target_school}."
                           "My research experience is: '{template_research_experience} '."
                           "My Internship experience is:'{template_internship_experience}'.")
    llm_output_format = ("Name provided: Yes(If yes, write the name)/No"
                         "School provided: Yes(If yes, write the School)/No"
                         "GPA provided: Yes(If yes, write the GPA)/No"
                         "Major provided: Yes(If yes, write the Major)/No"
                         "Target University provided: Yes(If yes, write the Targeted School)/No"
                         "Research Experience provided: Yes(If yes, write the Research Experience)/No"
                         "Internship Experience provided: Yes(If yes, write the Internship Experience)/No'")

    user_input_template = PromptTemplate(
        input_variables=["template_name", "template_gpa", "template_major", "template_university",
                         "template_research_experience",
                         "template_internship_experience", "template_target_school"],
        template=basic_info_template,
    )

    # Format the template as a string
    user_input = user_input_template.format(
        template_name="Rain",
        template_gpa="3.9",
        template_major="Computer Science",
        template_university="Sichuan University",
        template_research_experience="I joined Professor Ming Hsuan Yang's Lab, where I worked on a project to help "
                                     "malfunctioning robots complete tasks. Specifically, I developed algorithms to "
                                     "improve fault tolerance in robotic systems. This aligns with UC Berkeley's "
                                     "'Introduction to Robotics' course, which will provide a structured foundation "
                                     "to further my understanding and skills in robotics. Additionally, I aim to join "
                                     "Berkeley’s Robotic Learning Lab to focus on advancing these algorithms for "
                                     "practical applications in surgical robotics.",
        template_internship_experience="At Microsoft, I worked on a project involving the development of a new "
                                       "machine learning model to improve the accuracy of their speech recognition "
                                       "software. This experience is highly relevant to UC Berkeley’s courses like "
                                       "Advanced Machine Learning' and 'Natural Language Processing', which will help "
                                       "me deepen my understanding of machine learning techniques and their "
                                       "applications."
        ,
        template_target_school="University of California, Berkeley"
    )
    print(user_input)
    # Pass the formatted string to HumanMessage
    messages = [
        SystemMessage(
            content="You are a PS consultant. Your task is to ensure that the information provided by the user about their personal statement is integrated. "
                    f"Notice, the input has a template format {basic_info_template}, and it has variables like template_name, template_gpa, template_university,template_research_experience, template_internship_experience, template_targeted_school, etc."
                    "Below is three steps you must follow to judge whether the information is comprehensive, and you should also print the three corresponding contents out in the following sequence:"
                    f"First: you should judge whether user has completed all the input_variables (variables in the template), if the user hasn't completed any variable, please point it out in the following format:{llm_output_format}"
                    "Second: if you determine that there is any information in Step one that the user has not provided, indicated by a 'No', please output in the following format: 'You need to provide the following information: ['xxx', ....]'. If the user has provided every information neede, just output:'You need to provide the following information: []'"
                    "Third: if you determine that there is any information in Step one that the user has not provided, read the other information he/she provided first. Then, you should think step by step and suggest him with an example with these information, and tell the user why you think the example you provide such an example; if the user has provided all the information, please say: All information needed is provided and valid!"

        ),
        HumanMessage(content=user_input)
    ]

    response = llm.invoke(messages)  # GPT response
    print(response.content)
    # Find Step Three's content
    response = str(response.content)
    step_three_match = re.search(r'You need to provide the following information:\s*\[(.*?)\]', response)

    # Create a dictionary to store extracted information
    info_notprovided = {}

    if step_three_match:
        # Extract content and handle empty list case
        items = step_three_match.group(1).strip()
        if items:
            # If not empty, split by commas and remove spaces and quotes
            items = items.replace("'", "").replace("\\", "").split(', ')
            for item in items:
                info_notprovided[item] = None   # Initialize each item with None
        else:
            info_notprovided = {}  # Case for empty list

    # Create a dictionary to store user information
    user_info = {}

    # Use regular expressions to extract user information
    fields = [
        "Name provided",
        "School provided",
        "GPA provided",
        "Major provided",
        "Research Experience provided",
        "Internship Experience provided",
        "Target University provided"
    ]

    for field in fields:
        match = re.search(rf'{field}:\s*Yes\s*\((.*?)\)', response)
        if match:
            user_info[field.split()[0]] = match.group(1) # Only take the field name

    # Handle cases where information is provided as No
    for field in fields:
        if "No" in response and field in response:
            if f"{field}: No" in response:
                user_info[field.split()[0]] = None # Set as None

    # Store in the state dictionary
    state["user_info"] = user_info

    # print(state["user_info"])

    # Output results
    # print(info_notprovided)

    # Check for missing information
    missing_info = {key: value for key, value in info_notprovided.items() if value is None}

    while missing_info:
        for key in missing_info.keys():
            user_input = input(f"Please provide your {key} information: ")
            if user_input:
                state["user_info"][key] = user_input
                missing_info[key] = user_input
            else:
                print("You did not enter any information, please try again.")

        # Check if there is still missing information
        missing_info = {key: value for key, value in info_notprovided.items() if state["user_info"].get(key) is None}

    print("All information has been completed:", state["user_info"])

    example_experience = "The Meng in EECS program at UC Berkeley, focusing on Robotics and Embedded Software, aligns perfectly with my goal of innovating in medical robotics. The Introduction to Robotics course will provide a structured foundation in robotics, which I currently explore without formal study. This systematic approach will enhance my ability to develop advanced robotic systems. Courses like Advanced Control Systems will deepen my understanding of controllability and observability, crucial for ensuring robots can operate effectively under challenging conditions. I aim to design control strategies that allow robots to function despite joint failures, using observability to infer internal states from external data. Additionally, the Algorithmic Human-Robot Interaction course will help me design robots that effectively communicate with patients and healthcare professionals. I am also eager to join Berkeley's Robotic Learning Lab to focus on surgical robotics, aspiring to create robots capable of remote surgeries. After graduation, I envision working at leading robotics companies like Boston Dynamics, contributing to innovations that enable remote medical procedures, including diagnostics and treatments."

    while not state["input_validate"]:
        # print(state['user_info']['Internship'])
        messages = [
            SystemMessage(
                content="You are a  PS consultant. Your task is to decide whether the information provided by the user meets the four following standards. Here are the standards:"
                        "First, check whether the user's research experience and internship experience is relavant with his major."
                        "Second, check whether the user has given details of what he/she did in his research or internship. For example, if the user only said that he has done a job, then he/she violates this standard, but if he said that he has done a visualization job, that meets the standard."
                        f"Third, Check whether the user mentioned the relevance of these experiences to their target school({state['user_info']['Target']})'s curriculums (courses) or labs (only mentioned labs or curriculums (courses) is OK) in both the research experience and internship experience sections, aside from the project introduction. For instance, this should be a validated input:{example_experience}."
                        "Four, If the user violates any of the top three standards, you must provide an example to show how to meet those standards."
                        "After acknowledging and memorizing the four standards, I am going to teach you how to generate your response: If the research experience doesn't meet any of the the standards, please point out the standard it violets and explain the reasons. At the last of your response, you should print your judgement in this format: 'Research: Validate (or Invalidate); Internship: Validate (or Invalidate).' "
                        "Here is a response example:\n"
                        "'The research experience meets all the standards:\n"
                        "1. The research experience is relevant to the user's major.\n"
                        "2. The user has provided details about what they did in their research, specifically developing algorithms to improve fault tolerance in robotic systems.\n"
                        "3. The user mentioned the relevance of this experience to UC Berkeley's curriculum and labs, referencing the 'Introduction to Robotics' course and the Robotic Learning Lab.\n"
                        "The internship experience violates the second and third standards:\n"
                        "1. The internship experience is relevant to the user's major.\n"
                        "2. The user has not provided specific details about what they did during the internship, only stating that they worked on a project involving the development of a new machine learning model.\n"
                        "3. The user did not mention the relevance of this experience to UC Berkeley's curriculums or labs.\n"
                        "To meet these standards, the user could provide more detail on their tasks and specify the relevance to UC Berkeley's offerings. For example:\n"
                        "At Microsoft, I developed a new machine learning model for predictive analytics, which involved data preprocessing, feature selection, and model evaluation. This experience aligns with UC Berkeley's 'Machine Learning' course, where I can deepen my understanding of advanced algorithms and their applications. Additionally, I am keen to join the Berkeley AI Research Lab to contribute to groundbreaking work in machine learning.\n"
                        "Research: Validate; Internship: Invalidate.'\n"
                        "\n\n"
                        "Now keep in mind the four standards and the way to generate your response, think step by step to judge this user input information and MUST output your judgement in this format, ';' must also be included: 'Research: Validate (or Invalidate); Internship: Validate (or Invalidate).':"

            ),
            HumanMessage(content=
                         f"Research Experience: {state['user_info']['Research']}."
                         f"Internship Experience: {state['user_info']['Internship']}.")  # 使用格式化后的字符串
        ]

        is_validate = llm.invoke(messages)  # GPT response
        print(is_validate.content)

        # Match pattern to allow spaces and all combinations
        pattern_is_validate = r"Research:\s*(Validate|Invalidate)\s*;\s*Internship:\s*(Validate|Invalidate)"

        # Extract the matching result
        match_is_validate = re.search(pattern_is_validate, is_validate.content)

        # Check the extracted result and update the status
        if match_is_validate:
            research_status, internship_status = match_is_validate.groups()
            # Print the extracted part
            print("Validate?", match_is_validate.group())
            # Set to True if both are Validate
            if research_status.lower() == "validate" and internship_status.lower() == "validate":
                state["input_validate"] = True
            elif research_status.lower() == "invalidate" or internship_status.lower() == "invalidate":
                if research_status.lower() == "invalidate":
                    research_status_input = input(
                        f"Please re-enter your research experience information to meet the requirements: ")
                    state['user_info']['Research'] = research_status_input
                if internship_status.lower() == "invalidate":
                    internship_status_input = input(
                        f"Please re-enter your internship experience information to meet the requirements: ")
                    state['user_info']['Internship'] = internship_status_input
            else:
                raise ValueError("INVALID SEARCH RESULT!")
        else:
            raise ValueError("NO VALIDATION INFO DETECTED!")

    return state


# Define the domain expert step
def domain_expert_step(state: State) -> dict:
    print("Domain_Expert_Step:")
    school = state["user_info"].get("School")
    major = state["user_info"].get("Major")
    target_school = state["user_info"].get("Target")

    if not school or not major or not target_school:
        print("School or Major information is missing.")
        raise ValueError("School, Major, or Target School information is missing.")

    # Construct queries
    query_school = f"{school}"
    query_major = f"{major}"
    query_target = f"{target_school}"

    # Retrieve related information from Wikipedia
    # Retrieve school information
    try:
        summary_school = wikipedia.summary(query_school)
        # Add the retrieved information to the state
        state["domain_data"]["school"][query_school] = summary_school
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation error: {e}")
    except wikipedia.exceptions.PageError:
        print("Page not found for the user's school.")

    # Retrieve major information
    try:
        summary_major = wikipedia.summary(query_major)
        # Add the retrieved information to the state
        state["domain_data"]["major"][query_major] = summary_major
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation error: {e}")
    except wikipedia.exceptions.PageError:
        print("Page not found for the user's major.")

    # Retrieve target school information
    try:
        summary_target = wikipedia.summary(query_target)
        # Add the retrieved information to the state
        state["domain_data"]["target"][query_target] = summary_target
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation error: {e}")
    except wikipedia.exceptions.PageError:
        print("Page not found for the user's target school.")

    # print(state["domain_data"])

    return state


# Define the sophisticated writer step
def sophisticated_writer_step(state: State) -> dict:
    print("Sophisticated_Writer_Step:")
    input_info_template = (
        "The user's personal information is: {usr_info}. And here are some additional relevant domain information to help you understand the user-provided profile:"
        "The detail user's school: {det_school},"
        "The detail user's major: {det_major},"
        "The detail user's target school: {det_tar_school}."
    )

    user_input_template = PromptTemplate(
        input_variables=["usr_info", "extra_info"],
        template=input_info_template,
    )

    # Format the template into a string
    input_info = user_input_template.format(
        usr_info=state["user_info"],
        det_school=state["domain_data"]["school"],
        det_major=state["domain_data"]["major"],
        det_tar_school=state["domain_data"]["target"]
    )
    # print(input_info)

    # If already judged
    if state["is_judged"]:
        messages = [
            SystemMessage(
                content="Using the following user-provided personal information and relevant domain knowledge, please compose a Personal Statement for the student. "
                        "The statement should be clear and logically structured, demonstrating a strong narrative flow and using a formal, professional tone to highlight the user’s academic background, research skills, and career aspirations. Do not forget to associate the user's research and internship experience with their target school's courses or labs (already mentioned in the experience information)! "
                        "Focus on storytelling techniques to make the content engaging and impactful. "
                        "Here is a draft you gave last time and the reason why it does not meet the standard:"
                        f"{state['judge_result']}"
                        "\n\n"
                        "You should learn from the judgment and write a better PS with the given information. Also, keep in mind the requirements I mentioned at the beginning of this prompt:"
            ),
            SystemMessage(content=input_info)  # Use the formatted string
        ]

        response = llm.invoke(messages)  # GPT response
        state["ps_draft"] = str(response.content)
        print(state["ps_draft"])
    else:
        # If writing for the first time
        messages = [
            SystemMessage(
                content="Using the following user-provided personal information and relevant domain knowledge, please compose a Personal Statement for the student. "
                        "The statement should be clear and logically structured, demonstrating a strong narrative flow and using a formal, professional tone to highlight the user’s academic background, research skills, and career aspirations. Do not forget to associate the user's research and internship experience with their target school's courses or labs (already mentioned in the experience information)! "
                        "Focus on storytelling techniques to make the content engaging and impactful."
            ),
            SystemMessage(content=input_info)  # Use the formatted string
        ]

        response = llm.invoke(messages)  # GPT response
        state["ps_draft"] = str(response.content)
        print(state["ps_draft"])

    return state


# Define the judge step
def judge_step(state: State) -> dict:
    print("Judge_Step:")
    if not state["output_validate"]:
        messages = [
            SystemMessage(
                content="You are a PS judge. Your role is to determine whether the user’s personal statement meets the following four standards. Here are the standards you should apply:"
                        "First, Mention of Experience: Assess whether the user's research and internship experience are mentioned in the PS."
                        f"Second, Connection to Target School {state['user_info']['Target']}: Verify whether the user connects their experiences to relevant courses or labs or research opportunities that is related to target school:{state['user_info']['Target']}. The mention can be general (e.g., naming courses or labs), but it should show awareness of the school’s offerings."
                        f"Third, Mention of Motivation : Verify whether the PS has mentioned the user's motivation, the reason he applys for the target school:{state['user_info']['Target']}."
                        "Fourth, Constructive Examples: If any of the above standards are not met, you must provide an example that demonstrates how to meet the respective standard."
                        "After reviewing these standards, generate your response according to these guidelines:"
                        "Identify which standards the user’s personal statement meets or violates and provide explanations for each."
                        "If the personal statement fails to meet any standard, provide a specific example illustrating how to meet it."
                        "Conclude your response with a validation summary in the following format at the end of your reponse: 'Personal Statement: Validate (or Invalidate)' Ensure that the judgment is separated by a semicolon and follows this structure."
                        "Here is a response example:"
                        "**Assessment of Personal Statement**"
                        "1. **Mention of Experience: Validate**"
                        "The personal statement clearly mentions the user's research and internship experience. The user details their involvement in a research project at Professor Ming Hsuan Yang's Lab, focusing on fault tolerance in robotic systems. Additionally, the user describes their internship at Microsoft, where they worked on speech recognition software and developed a machine learning model."
                        "2. **Connection to Target School: Validate**"
                        "The user effectively connects their experiences to relevant courses and labs at the University of California, Berkeley. They mention specific courses like 'Introduction to Robotics,' 'Advanced Machine Learning,' and 'Natural Language Processing.' They also express interest in joining Berkeley’s Robotic Learning Lab, showing awareness of the school's offerings."
                        "3. **Mention of Motivation: Validate**"
                        "The user's motivation for applying to the University of California, Berkeley, is clearly articulated. They cite the institution’s excellence in research and innovation, the alignment of its courses with their prior research, and the potential for significant advancements in medical technology as key motivators for their application."
                        "4. **Constructive Examples: Not needed**"
                        "Since all standards are met, there is no need to provide constructive examples."
                        "**Validation Summary:**"
                        "Research: Validate; Internship: Validate."
                        "Remember to adhere to the standards and the output format strictly. Now, let's think step by step, using the four standards and response method (example, summary), evaluate the user's personal statement input and provide your assessment."
            ),
            SystemMessage(content=f"The PS draft is: {state['ps_draft']}")
        ]

        is_validate = llm.invoke(messages)  # GPT response
        print(is_validate.content)
        state["is_judged"] = True
        state["judge_result"] = is_validate.content

        pattern_is_validate = r"Personal\s*Statement\s*:\s*(Validate|Invalidate)"

        match_is_validate = re.search(pattern_is_validate, is_validate.content)

        if match_is_validate:
            ps_draft_status = match_is_validate.groups()
            print("Validate?", match_is_validate.group())
            if ps_draft_status[0].lower() == "validate":
                state["output_validate"] = True
        else:
            raise ValueError("NO VALIDATATION INFO DETECTED!")
    return state


# Define the judge_validate function
def judge_validate(state: State) -> str:
    return "yes" if state["output_validate"] else "no"


# Initialize state graph
graph = StateGraph(State)

# Adding nodes
graph.add_node("consultant_step", consultant_step)
graph.add_node("domain_expert_step", domain_expert_step)
graph.add_node("sophisticated_writer_step", sophisticated_writer_step)
graph.add_node("judge_step", judge_step)

# Setting entry and finish points
graph.set_entry_point("consultant_step")
graph.add_edge("consultant_step", "domain_expert_step")
graph.add_edge("domain_expert_step", "sophisticated_writer_step")
graph.add_edge("sophisticated_writer_step", "judge_step")

# Adding conditional edges for the judge step
graph.add_conditional_edges(
    "judge_step",
    judge_validate,
    {
        "yes": END,
        "no": "sophisticated_writer_step",
    }
)

# Compile the graph
compiled_graph = graph.compile()

# Initial state with user_info and other data included
initial_state = {
    "input_validate": False,
    "output_validate": False,
    "user_info": {},
    "ps_draft": "",
    "domain_data": {
        "school": {},
        "major": {},
        "target": {}
    },
    "judge_result": "",
    "is_judged": False
}

# Running the compiled graph
result = compiled_graph.invoke(initial_state)
print(f"The PS draft is: {result['ps_draft']}")
