import os
import autogen
import openai

from autogen import config_list_from_json
from dotenv import load_dotenv
from google_search import search
from web_scraper import scrape

# Get API key
load_dotenv()

config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST.json")
openai.api_key = os.getenv("OPENAI_API_KEY")


def research(query):
    llm_config_researcher = {
        "functions": [
            {
                "name": "search",
                "description": "google search for relevant information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Google search query",
                        }
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "scrape",
                "description": "Scraping website content based on url",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                           "type": "string",
                            "description": "Website url to scrape",
                        }
                    },
                    "required": ["url"],
                },
            },
        ],
        "config_list": config_list}

    researcher = autogen.AssistantAgent(
        name="researcher",
        system_message="Research about a given query, collect as many information as possible, and generate detailed "
                       "research results with loads of technique details with all reference links attached; Add "
                       "TERMINATE to the end of the research report;",
        llm_config=llm_config_researcher,
    )

    user_proxy = autogen.UserProxyAgent(
        name="User_proxy",
        code_execution_config={"last_n_messages": 2, "work_dir": "coding"},
        is_termination_msg=lambda x: x.get("content", "") and x.get(
            "content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        function_map={
            "search": search,
            "scrape": scrape,
        }
    )

    user_proxy.initiate_chat(researcher, message=query)

    # set the receiver to be researcher, and get a summary of the research report
    user_proxy.stop_reply_at_receive(researcher)
    user_proxy.send(
        "Give me the research report that just generated again, return ONLY the report & reference links",
        researcher)

    # return the last message the expert received
    return user_proxy.last_message()["content"]


# Define write content function
def write_content(research_material, topic):
    editor = autogen.AssistantAgent(
        name="editor",
        system_message="You are a senior editor of an AI blogger, you will define the structure of a short blog post "
                       "based on material provided by the researcher, and give it to the writer to write the blog post",
        llm_config={"config_list": config_list},
    )

    writer = autogen.AssistantAgent(
        name="writer",
        system_message="You are a professional AI blogger who is writing a blog post about AI, you will write a short "
                       "blog post with no less than 1000 words, based on the structured provided by the editor, "
                       "and feedback from reviewer; After 2"
                       "rounds of content iteration, add TERMINATE to the end of the message",
        llm_config={"config_list": config_list},
    )

    reviewer = autogen.AssistantAgent(
        name="reviewer",
        system_message="You are a world class hash tech blog content critic, you will review & critic the written "
                       "blog and provide feedback to writer.After 2 rounds of content iteration, add TERMINATE to the "
                       "end of the message",
        llm_config={"config_list": config_list},
    )

    user_proxy = autogen.UserProxyAgent(
        name="admin",
        system_message="A human admin. Interact with editor to discuss the structure. Actual writing needs to be "
                       "approved by this admin.",
        code_execution_config=False,
        is_termination_msg=lambda x: x.get("content", "") and x.get(
            "content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
    )

    groupchat = autogen.GroupChat(
        agents=[user_proxy, editor, writer, reviewer],
        messages=[],
        max_round=20)
    manager = autogen.GroupChatManager(groupchat=groupchat)

    user_proxy.initiate_chat(
        manager, message=f"Write a blog about {topic}, here are the material: {research_material}")

    user_proxy.stop_reply_at_receive(manager)
    user_proxy.send(
        "Give me the blog that just generated again, return ONLY the blog, and add TERMINATE in the end of "
        "the message",
        manager)

    # return the last message the expert received
    return user_proxy.last_message()["content"]


# Define content assistant agent
llm_config_content_assistant = {
    "functions": [
        {
            "name": "research",
            "description": "research about a given topic, return the research material including reference links",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The topic to be researched about",
                    }
                },
                "required": ["query"],
            },
        },
        {
            "name": "write_content",
            "description": "Write content based on the given research material & topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "research_material": {
                        "type": "string",
                        "description": "research material of a given topic, including reference links when "
                                       "available",
                    },
                    "topic": {
                        "type": "string",
                        "description": "The topic of the content",
                    }
                },
                "required": ["research_material", "topic"],
            },
        },
    ],
    "config_list": config_list}

writing_assistant = autogen.AssistantAgent(
    name="writing_assistant",
    system_message="You are a writing assistant, you can use research function to collect latest information about a "
                   "given topic, and then use write_content function to write a very well written content. After 2 "
                   "rounds of content iteration, add TERMINATE to the end of the message",
    llm_config=llm_config_content_assistant,
)

user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    human_input_mode="NEVER",
    system_message="TERMINATE when writing_assistant sends you a message with TERMINATE in",
    function_map={
        "write_content": write_content,
        "research": research,
    }
)

message_1 = ("Research on the latest crypto market performance, and recommend the best performing crypto currencies. "
             "TERMINATE to the end of the message")

# message_2 = ("Write a blog post about the crypto performance this week. List all the best performing crypto
# currencies " "as well. Do not list URLs but only the details ! Add TERMINATE to the end of the message")

user_proxy.initiate_chat(
    writing_assistant, message=message_1)
