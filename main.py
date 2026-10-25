from itertools import chain
from urllib import response

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()



def main():
    print("Hello from langchain-course!")
    info = """ Barack Hussein Obama II[a] (born August 4, 1961) is an American politician who served as the 44th president of the United States from 2009 to 2017. A member of the Democratic Party, he was the first African American to serve as president. Obama represented Illinois in the United States Senate from 2005 to 2008 and served as an Illinois state senator from 1997 to 2004.

    Born in Honolulu, Obama graduated from Columbia University in 1983 and later worked as a community organizer in Chicago. In 1988, Obama enrolled at Harvard Law School, where he was the first Black president of the Harvard Law Review. He became a civil rights attorney and an academic, teaching constitutional law at the University of Chicago Law School from 1992 to 2004. In 1996, Obama was elected to represent the 13th district in the Illinois Senate. In 2000, Obama ran for Illinois's 1st congressional district, but came in second to the incumbent Representative, Bobby Rush. Following his loss in the House race, he kept his position he held in the State Senate until 2004, when he successfully ran for the U.S. Senate. In the 2008 presidential election, after a close primary campaign against Hillary Clinton, he was nominated by the Democratic Party for president. Obama selected Joe Biden as his running mate and defeated Republican nominee John McCain and his running mate Sarah Palin."""

    summary_template = """ 
    given the information {info} about a person I want you to create:
    1. A short story
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["info"], template=summary_template
    )

    llm=ChatOllama(temperature=0, model="gpt-oss:20b")
    chain=summary_prompt_template | llm
    print("INFO BEING SENT TO MODEL:")
    print(info)

    response = chain.invoke(input={"info": info})

    print(response.content)

if __name__ == "__main__":
    main()
