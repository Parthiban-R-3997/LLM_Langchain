from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.chains import SimpleSequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv
load_dotenv()

# Create an OpenAI LLM object
llm = OpenAI(temperature=0.6)
#print(llm('What is the capital of Newzealand'))

prompt= PromptTemplate(input_variables=['country'],template='tell me the capital of {country}')
#print(prompt.format(country='India'))
chain=LLMChain(llm=llm,prompt=prompt)
#print(chain.run('India'))


## SimpleSequentialChain
capital_prompt=PromptTemplate(input_variables=['country'], template='Please tell me the capital of the {country}')
capital_chain= LLMChain(llm=llm,prompt=capital_prompt)

famous_prompt=PromptTemplate(input_variables=['capital'], template='Please suggest some famous places in {capital}')
famous_chain= LLMChain(llm=llm,prompt=famous_prompt)

#simple_sequential_chain= SimpleSequentialChain(chains=[capital_chain,famous_chain])
#print(simple_sequential_chain.run('TamilNadu'))


##SequentialChain
capital_prompt=PromptTemplate(input_variables=['country'], template='Please tell me the capital of the {country}')
capital_chain= LLMChain(llm=llm,prompt=capital_prompt,output_key='capital')

famous_prompt=PromptTemplate(input_variables=['capital'], template='Please suggest some famous places in {capital}')
famous_chain= LLMChain(llm=llm,prompt=famous_prompt,output_key='places')

#simple_chain=SequentialChain(chains=[capital_chain,famous_chain],input_variables=['country'],output_variables=['capital','places'])
#print(simple_chain({'country':'United Kingdom'}))


#Chatmodels with ChatOpenAI
''' 3 important schemas
1. Human message (input)
2. System message (Default message)
3. AI message (output)'''

chatllm=ChatOpenAI(temperature=0.6,model='gpt-3.5-turbo')
#result=chatllm([SystemMessage(content='You are a cricket analyst'),HumanMessage(content='Please provide some information about cricket 50 over world cup')])
#print(result)

'''
Prompt Template + LLM + Output Parsers

If we want to modify any output of the LLM model before hand
'''

class commanseperatedoutput(BaseOutputParser):
    def parse(self,text:str):
        return text.strip().split(',')
    
system_template="Your are a helpful assistant. When the user given any input , you should generate 5 words synonyms in a comma seperated list"
human_template="{text}"
chatprompt=ChatPromptTemplate.from_messages([
    ("system",system_template),
    ("human",human_template)
])    

chat_chain=chatprompt|chatllm|commanseperatedoutput()
## Whenever we use chains give it in key-value pairs

print(chat_chain.invoke({"text":"Data Science"}))