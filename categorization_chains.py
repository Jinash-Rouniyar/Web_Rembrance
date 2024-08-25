import os
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import  ChatOpenAI
try:
    from langchain_groq import ChatGroq
    from langchain_anthropic import ChatAnthropic
except:
    print("Error importing groq/anthropic")
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import (
     PromptTemplate,
     SystemMessagePromptTemplate,
     HumanMessagePromptTemplate,
     ChatPromptTemplate,
 )

groq_api_key = os.environ.get('GROQ_API_KEY')
claude_api_key = os.getenv("ANTHROPIC_API_KEY")

if not claude_api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")


review_system_template_str = """
            You are an expert in the new digital SAT Reading and Writing section. Your task is to analyze given paragraphs and their associated questions, then categorize each question into one of the four main types: Information and Ideas, Craft and Structure, Expression of Ideas, or Standard English Conventions.
            When presented with a paragraph and question, you should:
            Carefully read the provided text and question.
            Analyze the nature of the question and what skills it's testing.
            Determine which of the four main categories the question falls under.
            Provide only the category name as your answer, without any additional explanation or commentary.
            
            Your response should be in the following format:
            [Category Name]
            
            Important guidelines for categorization:
            1) Information and Ideas:
            Questions about main ideas, details, evidence, and inferences
            Includes questions asking to "logically complete the text" based on given information
            Requires using, locating, interpreting, inferring, or evaluating information from the text
            
            2) Craft and Structure:
            Questions about word meanings in context, text purpose, and connections between texts
            Focuses on how the text is constructed and its rhetorical elements

            3) Expression of Ideas:
            Questions specifically about revising or improving the text
            Involves synthesizing ideas or making effective transitions within the given text

            4)Standard English Conventions:
            Questions about grammar, usage, punctuation, and sentence structure
            Focuses on editing the text to conform to standard written English

            Provide only the main category name as your answer, without any explanations or additional information.
"""

review_system_prompt = SystemMessagePromptTemplate(
     prompt=PromptTemplate(
        template=review_system_template_str
     )
 )
review_human_prompt = HumanMessagePromptTemplate(
     prompt=PromptTemplate(
         input_variables=["question"], template="{question}"
     )
 )

messages = [review_system_prompt, review_human_prompt]

review_prompt_template = ChatPromptTemplate(
     input_variables=["question"],
     messages=messages,
 )
output_parser = StrOutputParser()

#GPT Answer Chain
gpt_chat_model = ChatOpenAI(model="gpt-4o", temperature=0)
gpt_question_chain = (
    {"question": RunnablePassthrough()}
    | review_prompt_template
    | gpt_chat_model
    | StrOutputParser()
)
#Groq Answer Chain
groq_chat_model = ChatGroq(
    api_key=groq_api_key, 
    model_name='mixtral-8x7b-32768',
    temperature = 0
)
groq_question_chain = (
    {"question": RunnablePassthrough()}
    | review_prompt_template
    | groq_chat_model
    | StrOutputParser()
)
#Claude Answer Chain
claude_chat_model = ChatAnthropic(model="claude-3-sonnet-20240620", 
                           temperature=0, 
                           max_tokens=1024,
                           timeout=None,
                           max_retries=2,
                           api_key = claude_api_key
                           )

claude_question_chain = (
    {"question": RunnablePassthrough()}
    | review_prompt_template
    | claude_chat_model
    | StrOutputParser()
)

