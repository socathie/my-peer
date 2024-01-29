import streamlit as st
import os
import promptlayer
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import AzureChatOpenAI
from langchain_community.callbacks import PromptLayerCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import time

os.environ["AZURE_OPENAI_API_KEY"] = st.secrets["AZURE_OPENAI_API_KEY"]
os.environ["AZURE_OPENAI_ENDPOINT"] = st.secrets["AZURE_OPENAI_ENDPOINT"]
promptlayer.api_key = st.secrets["PROMPTLAYER_API_KEY"]

st.set_page_config(page_title="My Peer")

if "disabled" not in st.session_state:
    st.session_state["disabled"] = False

# parameters
avatars = {
    "human": "💬",
    "ai": "🧑🏻‍💻"
}

if "id" not in st.session_state:
    st.session_state["id"] = st.query_params.get("id", "error")
    if st.session_state["id"] == "error":
        st.error("Please provide a participant ID in the URL.")
        st.stop()
        
tag = st.session_state["id"]
# tag = st.sidebar.text_input("Participant ID", value=st.session_state["id"], disabled=True)

# developer mode toggle
# developer_mode = st.sidebar.checkbox("Developer Mode", value=False) # TODO: comment out this line in production

# if developer_mode:
#     reading_cpm = st.sidebar.number_input("Reading speed (characters per minute)", value=200)
#     typing_cpm = st.sidebar.number_input("Typing speed (characters per minute)", value=50)
#     first_msg = st.sidebar.text_input("First message", value="告訴我一件最近不開心的事吧！")
#     system_msg = st.sidebar.text_area("System message", value="You are a therapist. You want to keep anonymous. Do not mention that you are an AI. Only use Traditional Chinese to respond.", height=300)
#     # button to update and reload
#     if st.sidebar.button("Update"):
#         st.session_state.clear()
#         st.session_state["disabled"] = False
# else:
reading_cpm = 200
typing_cpm = 50

first_msg = "告訴我一件最近不開心的事吧！"
system_msg = "You are a therapist. You want to keep anonymous. Do not mention that you are an AI. Only use Traditional Chinese to respond."

reading_cps = reading_cpm/60
typing_cps = typing_cpm/60

llm = AzureChatOpenAI(
    openai_api_version="2023-12-01-preview",
    azure_deployment="gpt-4",
    model="gpt-4",
    model_version="1106-Preview",
    callbacks=[PromptLayerCallbackHandler(pl_tags=[tag])]
)

msgs = StreamlitChatMessageHistory()

memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs, return_messages=True)
if len(msgs.messages) == 0:
    msgs.add_ai_message(first_msg)

system_msg_template = SystemMessagePromptTemplate.from_template(template=system_msg)

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=memory, prompt=prompt_template, llm=llm, verbose=True)

for msg in msgs.messages:
    st.chat_message(msg.type, avatar=avatars[msg.type]).write(msg.content)

def disable():
    st.session_state["disabled"] = True

typing_html = """typing
<span class="dot-one"> .</span>
<span class="dot-two"> .</span>
<span class="dot-three"> .</span>

<style>
span[class^="dot-"]{
  opacity: 0;
}
.dot-one{
  animation: dot-one 3s infinite linear
}
.dot-two{
  animation: dot-two 3s infinite linear
}
.dot-three{
  animation: dot-three 3s infinite linear
}
@keyframes dot-one{
  0%{
    opacity: 0;
  }
  15%{
    opacity: 0;
  }
  25%{
    opacity: 1;
  }
  100%{
    opacity: 1;
  }
}

@keyframes dot-two{
  0%{
    opacity: 0;
  }
  25%{
    opacity: 0;
  }
  50%{
    opacity: 1;
  }
  100%{
    opacity: 1;
  }
}

@keyframes dot-three{
  0%{
    opacity: 0;
  }
  50%{
    opacity: 0;
  }
  75%{
    opacity: 1;
  }
  100%{
    opacity: 1;
  }
}
</style>
"""

if prompt := st.chat_input(on_submit=disable, disabled=st.session_state.disabled):
    st.chat_message("human", avatar=avatars["human"]).write(prompt)

    start = time.time()
    response = conversation.invoke(input=prompt, history=msgs.messages)
    end = time.time()

    # time delay to simulate reading
    time.sleep(max(len(prompt)/reading_cps - (end-start), 0))

    # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
    with st.chat_message("ai", avatar=avatars["ai"]):
        st.write(typing_html, unsafe_allow_html=True)
        # time delay to simulate typing
        time.sleep(len(response)/typing_cps)
        st.write(response)
    st.session_state["disabled"] = False
    # hacky way to reenable chat input
    st.rerun()
