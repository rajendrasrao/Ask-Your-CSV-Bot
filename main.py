from langchain_google_genai  import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatMessagePromptTemplate
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
import streamlit 
import pandas as pd
import streamlit as st

def main():
        load_dotenv()
        st.set_page_config(
            page_title="this is my page",
            page_icon=":books:"
        )
        uploaded_file = st.file_uploader("Choose a file",type=["csv"])
       
        st.title("CSV question answer Chatbot")
        st.subheader("Uncover Insights from your Data!")
        
        st.markdown("""
        This chatbot was created to answer questions from a csv uploaded by you.
        Ask a question and the chatbot will respond with appropriete Analysis.
        """)
        if uploaded_file is not None:
          with open(uploaded_file.name, mode='wb') as w:
                w.write(uploaded_file.getvalue())
       
          df=pd.read_csv(uploaded_file.name)
          st.write(df.head())
          user_question=st.text_input("ask your question")
          print(user_question)
          if user_question is not None:
            
             llm=ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0.0)
        
             agent=create_csv_agent(llm=llm,path=uploaded_file
                               ,verbose=True,
                        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,allow_dangerous_code=True)
             #print(agent.agent.llm_chain.prompt.template)
             answer = agent.run(user_question)
             st.write("here is answer")
             st.write(answer)
    

    

if __name__=="__main__":
    main()


