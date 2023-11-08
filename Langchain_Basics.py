from  langchain import HuggingFaceHub
from dotenv import load_dotenv
load_dotenv()
llm=HuggingFaceHub(repo_id='google/flan-t5-large',model_kwargs={'temperature':0.6,'max_length':64})

output=llm.predict('What is the capital of United Kingdom')
print(output)

