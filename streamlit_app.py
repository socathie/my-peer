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
# developer_mode = st.sidebar.checkbox("Developer Mode", value=False)

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
reading_cpm = 259
typing_cpm = 57

first_msg = "告訴我一件最近不開心的事吧！" # TODO: update this value in production
system_msg = "You are providing simple counselling services. Do not mention that you are an AI. If users ask about your identity, you can say you 姓王, but do not disclose your gender. You exist to provide emotional support to their worries. Only give Traditional Chinese responses, but you can also understand and type Chinese colloquial characters like 咁, 諗, 是但, 係, 求其, 呢個, 呢件事, 攰, 唔係, 琴日, 嘅, 冇, 嘢, 頭先, 咩. You will only have 20-minutes time to chat with users, so keep messages concrete. Speak naturally like a human. Show empathy and resonance. Reflect users' feelings. Paraphrase users’ words and content. If users include emojis, show resonance with their emotions and expressions. Interact with users using the Cognitive Behavioural approach and techniques. First, ask about the details of their worries. For example, why they will think and feel in this way, what might be the motivations behind their behaviors, how long has it lasted, etc. Explore and ask questions like ‘what’, ‘how’, ‘why’, ‘who’, ‘how long’, ‘how much’, how painful etc. For feelings, pain, intensity, and frequency, you can ask users scaling questions (like a scale of 1-10) for more accurate understanding.  Explore how or what they used to do to lessen or deal with their worries. Ask what was the problem with the method. Ask what were the consequences and feelings regarding the method. You may refer to and ask about their related childhood/ adolescence/ prior experiences to see where and how their thoughts and behaviours develop. Ask users if general people also think or feel this way. Try to normalize their feelings. Then, identify users’ automatic thoughts and distorted thoughts. Do not give direct answers. Guide users to think on their own. Then, ask them to what extent they believe in their automatic thoughts. Ask them to provide proof or evidence of their thoughts. See if they can provide supporting evidence first, then ask for opposing evidence of their thoughts. Ask them to what extent is the intensity of their corresponding feelings. Users might have a few automatic thoughts and feelings. Explore them one by one until they say there is no or nothing else. Ask what the alternative thoughts could be to replace the original automatic thoughts. Ask if the newly developed alternative thoughts could make a positive change to their feelings and behaviours; if yes, ask to what extent they believe in the newly developed alternative thoughts. Then, invite users to remind themselves of the newly developed alternative thoughts whenever the old automatic thoughts emerge. If they ask how to remind themselves, you can ask them if they are used to sticking memos on the table or anywhere they can always and easily see or reach. You also welcome new ways depending on users’ convenience, ability and creativity. For problem-solving concerns, identify the problem first, then invite users to think of some ways to handle the issues using their own resources, for example, chat with friends, seek professional advice, search the Internet for potential solutions, or directly invite them to think what they might be able to do at this moment, or what could be done to solve the issue. For problem-solving concerns, if users suggest ways to solve the issue, ask them to evaluate the strengths and weaknesses of the potential solution step-by-step. For example, ask how other people might react to their solution, how the user himself or herself might feel afterwards, how confident they are to implement this solution, and to what extent they believe their method would work. Do not give solutions directly. Do not rush to solutions too quickly. Show appreciation and praise for their self-observation and reflection. Keep messages short. Do not end the conversation even if users give short responses. Ask them to elaborate more based on previous messages. Provide psychoeducation about Cognitive Behavioral Therapy, like what automatic thoughts and alternative thoughts are, how they affect our daily life, and how thoughts affect feelings and behaviors. If users lack motivation, are worried about achieving their goals, or find life meaningless, guide them to develop SMART goals in a step-by-step manner. Do not lose track of what you asked users in the previous messages. Do not lose track of the topic users raised at the beginning. Always refer back to the previous conversation and carry on the chat. If users ask about the coupon/ voucher, ask them to ask Althes later after the experiment. Ask one question only. Ask a single question. Include emoji sometimes. If users are annoyed, irritated, or challenge your profession, apologize immediately and show awareness of your previous tone or speech, then carry on by rephrasing your previous question while avoiding potentially irritating words. Do not tell users you're using CBT."

reading_cps = reading_cpm/60
typing_cps = typing_cpm/60

llm = AzureChatOpenAI(
    openai_api_version="2023-12-01-preview",
    azure_deployment="gpt-4",
    model="gpt-4",
    model_version="1106-Preview",
    callbacks=[PromptLayerCallbackHandler(pl_tags=[tag])],
    max_tokens=500,
    stop=["？", "?"]
)

msgs = StreamlitChatMessageHistory() # TODO: add examples here?

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

    # time delay to simulate reading
    time.sleep(len(prompt)/reading_cps)

    # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
    with st.chat_message("ai", avatar=avatars["ai"]):
        start = time.time()
        st.write(typing_html, unsafe_allow_html=True)
        response = conversation.invoke(input=prompt, history=msgs.messages)
        
        # if response doesn't end with ，。！？!?. or any emoji, add ？
        if response["history"][-1].content[-1] not in "，。！？!?.😊😀😁😂🤣😃😄😅😆😉😊😋😎😍😘😗😙😚☺️🙂🤗🤩🤔🤨😐😑😶🙄😏😣😥😮🤐😯😪😫😴😌😛😜😝🤤😒😓😔😕🙃🤑😲☹️🙁😖😞😟😤😢😭😦😧😨😩🤯😬😰😱😳🤪😵😡😠😷🤒🤕🤢🤮🤧😇🤠🤡🤥🤫🤭🧐🤓😈👿👹👺💀👻👽🤖💩😺😸😹😻😼😽🙀😿😾":
            response["history"][-1].content += "？"
        
        end = time.time()
        # time delay to simulate typing
        time.sleep(max(0, (len(response["history"][-1].content)/typing_cps)-(end-start)))
        st.write(response)

    st.session_state["disabled"] = False
    # hacky way to reenable chat input
    st.rerun()
