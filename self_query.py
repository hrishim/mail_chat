from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.chains.query_constructor.base import (
    get_query_constructor_prompt,
    load_query_constructor_runnable
)    

examples = [
    ( 
        "What is the email address of Gopal Srinivasan?",
        {
            "query": "foo",
            "filter": 'or(eq("to", "Gopal Srinivasan"),eq("sender", "Gopal Srinivasan"))'
            
        }
    ),

]

attribute_info = [
    AttributeInfo(
        name="to",
        description="The recipient of the email",
        type="string",
    ),
    AttributeInfo(
        name="sender",
        description="The sender of the email",
        type="string",
    ),
    AttributeInfo(
        name="subject",
        description="The subject of the email",
        type="string",
    ),
    AttributeInfo(
        name="date",
        description="The date the email was sent",
        type="string",
    ),
    AttributeInfo(
        name="content",
        description="The content of the email",
        type="string",
    ),
    AttributeInfo(
        name="x_gmail_labels",
        description="The labels of the email",
        type="string",
    ),
    AttributeInfo(
        name="x_gm_thrid",
        description="The thread ID of the email",
        type="string",
    ),
    AttributeInfo(
        name="inReplyTo",
        description="The ID of the email this is a reply to",
        type="string",
    ),
    AttributeInfo(
        name="references",
        description="The IDs of the emails this email references",
        type="string",
    ),
    AttributeInfo(
        name="message_id",
        description="The ID of the email",
        type="string",
    )
]

doc_contents_breaks_lark = "Detailed description of an email, the to field, sender field, subject field, date field, content field, x_gmail_labels field, x_gm_thrid field, inReplyTo field, references field, message_id field"
doc_contents= "Detailed description of an email"
prompt = get_query_constructor_prompt(
    doc_contents,
    attribute_info,
)
#print(prompt.format(query="{query}"))

# Initialize the conversational chain components
llm = ChatNVIDIA(
    base_url="http://0.0.0.0:8000/v1",
    model="meta/llama3-8b-instruct",
    temperature=0,
    max_tokens=1000,
    top_p=1.0
)

chain = load_query_constructor_runnable(llm, doc_contents, attribute_info)

def execute_chain_for_query(query):
    response = chain.invoke({"query": query})
    return response

print(execute_chain_for_query("When did I last sent email to Gopal Srinivasan?"))
print(execute_chain_for_query("What is Gopal Srinivasan's email?"))