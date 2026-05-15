from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatLiteLLM
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()  # Loads GROQ_API_KEY

# Define a simple calculator tool
def main():
    # Use Groq’s LLaMA-3 model through LiteLLM
    model = ChatLiteLLM(model="groq/llama-3.1-8b-instant", temperature=0)

    tools = []
    agent_executor = create_react_agent(model, tools)

    print("Welcome! I'm your AI assistant (Groq powered). Type 'quit' to exit.")
    print("You can ask me to perform calculations or chat with me.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quit":
            break

        print("\nAssistant:", end=" ")
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]}
        ):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    print(message.content, end="")
        print()

if __name__ == "__main__":
    main()
