from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import  ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import (
     PromptTemplate,
     SystemMessagePromptTemplate,
     HumanMessagePromptTemplate,
     ChatPromptTemplate,
 )

output_parser = StrOutputParser()
chat_model = ChatOpenAI(model="gpt-4o", temperature=0)

#Craft and Structure Bot
craft_system_template_str = """
           You are an expert in the new digital SAT Reading and Writing section. Your task is to analyze given questions and categorize them into one of the subcategories within the Craft and Structure category. 

            When presented with a question, you should:
            1. Carefully read the provided question.
            2. Analyze the nature of the question and what skills it's testing.
            3. Determine which of the three subcategories the question falls under: Vocabulary, Purpose, or Connection.
            4. Provide only the subcategory name as your answer, without any additional explanation or commentary.

            Your response should be in the following format:
            [Subcategory Name]

            Important guidelines for categorization:

            1) Vocabulary:
            Questions about word meanings in context or precise word choice
            Focuses on understanding and selecting appropriate vocabulary

            2) Purpose:
            Questions about text purpose, function, or structure
            Focuses on identifying the author's intent and text organization

            3) Connection:
            Questions comparing two texts or viewpoints
            Focuses on analyzing relationships between different passages

            Provide only the subcategory name (Vocabulary, Purpose, or Connection) as your answer, without any explanations or additional information.
"""

craft_system_prompt = SystemMessagePromptTemplate(
     prompt=PromptTemplate(
        template=craft_system_template_str
     )
 )
craft_human_prompt = HumanMessagePromptTemplate(
     prompt=PromptTemplate(
         input_variables=["question"], template="{question}"
     )
 )

craft_messages = [craft_system_prompt, craft_human_prompt]

craft_prompt_template = ChatPromptTemplate(
     input_variables=["question"],
     messages=craft_messages,
 )

craft_categorize_chain = (
    {"question": RunnablePassthrough()}
    | craft_prompt_template
    | chat_model
    | StrOutputParser()
)
#Information and Ideas Bot
information_idea_system_template_str = """You are an expert in the new digital SAT Reading and Writing section. Your task is to analyze given questions and categorize them into one of the subcategories within the Information and Ideas category. 

            When presented with a question, you should:
            1. Carefully read the provided question.
            2. Analyze the nature of the question and what skills it's testing.
            3. Determine which of the five subcategories the question falls under Main Ideas, Detail, Textual Evidence, Quantitative Evidence, or Inference
            4. Provide only the subcategory name as your answer, without any additional explanation or commentary.

            Your response should be in the following format:
            [Subcategory Name]

            Important guidelines for categorization:

            1) Main Ideas:
             Identified by keywords "main idea" or "central idea" in the question

            2) Detail:
             Uses phrases like "according to the passage," "the passage indicates," or "based on the passage, 
             what is true about..."
            
            3) Textual Evidence:
             Requires selecting information to support, illustrate, or weaken a claim
             Identified by phrases like "most strongly support," "most weakens," or "most effectively illustrates"

            4) Quantitative Evidence:
             Accompanied by a graph or table
             Asks for support for a claim using the provided data

            5) Inference:
              Uses keywords like "logically complete," "be inferred," "implies," or "what can be concluded"
              Correct answer is not explicitly stated in the passage

            Provide only the subcategory name (Main Idea, Detail, Textual Evidence, Quantitative Evidence, Inference) as your answer, without any explanations or additional information.

           """

information_idea_system_prompt = SystemMessagePromptTemplate(
     prompt=PromptTemplate(
        template=information_idea_system_template_str
     )
 )
information_idea_human_prompt = HumanMessagePromptTemplate(
     prompt=PromptTemplate(
         input_variables=["question"], template="{question}"
     )
 )

information_idea_messages = [information_idea_system_prompt, information_idea_human_prompt]

information_idea_prompt_template = ChatPromptTemplate(
     input_variables=["question"],
     messages=information_idea_messages,
 )

information_categorize_chain = (
    {"question": RunnablePassthrough()}
    | information_idea_prompt_template
    | chat_model
    | StrOutputParser()
)
#Expression of Ideas Bot
expression_ideas_system_template_str = """You are an expert in the new digital SAT Reading and Writing section. Your task is to analyze given questions and categorize them into one of the subcategories within the Expression of Ideas category. 

            When presented with a question, you should:
            1. Carefully read the provided question.
            2. Analyze the nature of the question and what skills it's testing.
            3. Determine which of the two subcategories the question falls under Synthesis or Transition
            4. Provide only the subcategory name as your answer, without any additional explanation or commentary.

            Your response should be in the following format:
            [Subcategory Name]

            Important guidelines for categorization:

            1) Synthesis:
             Questions that use bullet points instead of paragraph text
             Identifies a rhetorical aim and is often accompanied by which choice accomplishes the goal

            2) Transition:
            Questions about logical connections between parts of the text
            Often includes "logical transition" and asks for the most logical transition

            Provide only the subcategory name (Synthesis, Transition) as your answer, without any explanations or additional information.

           """

expression_ideas_system_prompt = SystemMessagePromptTemplate(
     prompt=PromptTemplate(
        template=expression_ideas_system_template_str
     )
 )
expression_ideas_human_prompt = HumanMessagePromptTemplate(
     prompt=PromptTemplate(
         input_variables=["question"], template="{question}"
     )
 )

expression_ideas_messages = [expression_ideas_system_prompt, expression_ideas_human_prompt]

expression_ideas_prompt_template = ChatPromptTemplate(
     input_variables=["question"],
     messages=expression_ideas_messages,
 )

expression_categorize_chain = (
    {"question": RunnablePassthrough()}
    | expression_ideas_prompt_template
    | chat_model
    | StrOutputParser()
)
#No sub-categories for Standard English Conventions

