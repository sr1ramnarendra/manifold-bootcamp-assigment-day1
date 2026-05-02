from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage,SystemMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


print("\n\nMultiple Query without Context -- Start")

resp1 = llm.invoke("We are building an AI system for processing medical insurance claims.")
#print(resp1)
print(resp1.content)

resp2 = llm.invoke("What are the main risks in this system?")
#print(resp2)
print(resp2.content)

print("\n\nMultiple Query without Context -- End")


print("\n\nMultiple Query with  Context -- Start")

messages = [SystemMessage(content="You are a senior AI architect reviewing production systems."),
            HumanMessage(content="We are building an AI system for processing medical insurance claims."),
            SystemMessage(content="What are the main risks in this system?")]

resp3 = llm.invoke(messages)
#print(resp3)
print(resp3.content)

print("\n\nMultiple Query with  Context -- End")


"""
Reflection:

1. Why did string-based invocation fail? LLM APIs are stateless, so for the send LLM call did not know the context
2. Why does message-based invocation work? With the messages list we are maintaining the context of the conversation
3. What would break in a production AI system if we ignore message contexthistory?  The AI system would not be able to maintain the context of the conversation, and would not be able to provide relevant responses. 
"""
