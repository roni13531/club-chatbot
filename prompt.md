# Codex Prompt: Curious Minds Club Chatbot (LangGraph + Supabase + Gradio)

## Goal
Build a Google Colab notebook (`club_chatbot.ipynb`) with a topic-locked LangGraph chatbot for an after-school center.
The user picks a club, and the bot talks only about that club's topic. Conversations are saved in a Supabase Postgres database through a `PostgresSaver` checkpointer, so they survive a Colab restart. Everything runs behind a Gradio `gr.Blocks` interface.

This extends the two class prompts (the chatbot with a checkpointer, and the Gradio interface). The graph stays the same one-node graph; what changes is where the state is saved, that the topic is part of the state, and the interface.

Keep it small: one node, no tools, no extra LLM calls, no loops inside the graph.

## Setup
- First cell:
  ```
  !pip -q install langgraph langchain langchain-openai langgraph-checkpoint-postgres "psycopg[binary]" gradio
  ```
- **Cell order matters:** Add a Secrets code cell immediately after the install cell and before the model cell:
  ```python
  import os
  from google.colab import userdata
  os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
  ```
  The key must be set before `init_chat_model(MODEL_NAME)` runs in the model cell, or model initialization fails with missing credentials.
- Add a markdown cell explaining that two Colab Secrets are required, both with Notebook access turned on:
  - `OPENAI_API_KEY`
  - `SUPABASE_DB_URL`: the Supabase **Session pooler** URI (IPv4; Colab cannot reach the IPv6 direct connection).
- Load both values only with `userdata.get`. Never use `getpass`. Never print, log or display the key, the password or the database URI, not even partly.

## Model
```python
MODEL_NAME = "openai:gpt-6-luna"
llm = init_chat_model(MODEL_NAME)
```
- Define `MODEL_NAME` once, in its own cell. Use `init_chat_model` from `langchain.chat_models`, as in class. Do not pass `temperature` (this model only accepts the default).
- Why this model: it is the low-cost model used in class, and answering short questions from teenagers does not need a large model.

## Topics
```python
TOPICS = ["Astronomy", "Cooking", "Football", "World Geography", "Python Programming"]
DEFAULT_TOPIC = "Astronomy"
```

## System message template
Store this text in a constant `SYSTEM_TEMPLATE`:
```text
You are the friendly chatbot of the {topic} club at Curious Minds, an after-school enrichment center for teenagers.

The current club is {topic}. Answer only questions about {topic}. Keep answers short, clear and suitable for teenagers.

Earlier messages in this conversation may belong to a different club. Only the current club counts: never continue or answer an earlier topic, even if it appears in the history.

If a question is not about {topic}, including a follow-up about an earlier club, do not answer it. Instead, say kindly that you can't help with that in the {topic} club, and suggest switching to the club that fits the question, if there is one. The clubs are: {clubs}.

Do not write essays, homework or assignments for other subjects, even if the request mentions {topic}.

Judge a question by what it actually asks, not by the words it contains. A question whose real subject is {topic} is on-topic even if it mentions another club; a question about another subject is off-topic even if it mentions {topic}.
```
Build the message with a helper:
```python
def build_system_message(topic: str) -> SystemMessage:
    return SystemMessage(SYSTEM_TEMPLATE.format(topic=topic, clubs=", ".join(TOPICS)))  # <-- the topic enters the system message here
```

## State (`TypedDict`)
| key | type | meaning |
|---|---|---|
| messages | `Annotated[list[AnyMessage], add_messages]` | the conversation (Human and AI messages only) |
| topic | `str` | the club the user chose; no reducer, so each new value replaces the old one |

## Node
Exactly one node, `chatbot`. It is the only place that calls the LLM, exactly once per user message:
```python
def chatbot(state: State) -> dict:
    system = build_system_message(state["topic"])
    reply = llm.invoke([system] + state["messages"])   # <-- the only LLM call
    return {"messages": [reply]}
```
- The `SystemMessage` is built fresh from `state["topic"]` on every call and is **never** stored in `messages`. The node returns only the reply.

## Graph
`START -> chatbot -> END`.

## Checkpointer (Supabase)
Use exactly this pattern:
```python
import os, psycopg
from psycopg.rows import dict_row
from google.colab import userdata
from langgraph.checkpoint.postgres import PostgresSaver

# OPENAI_API_KEY is already set in the Secrets cell at the top of the notebook
conn = psycopg.connect(userdata.get("SUPABASE_DB_URL"),
                       autocommit=True, row_factory=dict_row, prepare_threshold=0)
checkpointer = PostgresSaver(conn)   # <-- the checkpointer is created here
checkpointer.setup()                 # creates the checkpoint tables (safe to run again)
graph = builder.compile(checkpointer=checkpointer)
```
- No `InMemorySaver`. No Python variable (list, dict or global) that holds a conversation. The database is the only place conversations live.
- Add a short markdown note: the graph code is the same as with `InMemorySaver`; only the checkpointer line changes.

## Visualization
```python
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

## Graph test (before the interface)
Add this cell exactly, so the graph is tested on its own:
```python
config = {"configurable": {"thread_id": "test-1"}}

def ask(text, topic):
    result = graph.invoke({"messages": [HumanMessage(text)], "topic": topic}, config)
    print(f"[{topic}] {text}\n -> {result['messages'][-1].content}\n")

ask("How do I make pancakes?", "Cooking")                # answers
ask("Who won the 2022 World Cup?", "Cooking")            # refuses, suggests the Football club
ask("Who won the 2022 World Cup?", "Football")           # answers
ask("And how long do I fry each pancake?", "Football")   # refuses: old topic
```
Then a separate cell that prints the number of messages and the topic from `graph.get_state(config).values` (8 messages and `"Football"` on the first run; 16 on a second run, because the thread is saved).

## Conversations (threads)
- Each conversation has its own `thread_id = str(uuid.uuid4())`. Do not use `request.session_hash`.
- `list_conversations()` reads the list **from the database**, never from a Python variable:
  - Step 1: loop over `checkpointer.list(None)` and collect only the unique `thread_id`s, keeping the first time each one appears (newest first). Never call the checkpointer or graph inside this loop: `PostgresSaver.list` holds the connection lock while it yields, so another call can wait forever.
  - Step 2: after the list loop has finished, for each collected thread ID read `graph.get_state(config).values` and build a readable label: `"[{topic}] {first words of the first Human message}"` (about 40 characters, with `…` if cut).
  - Skip threads with no messages.
  - Return a list of `(label, thread_id)` pairs for a Gradio dropdown.
- `load_conversation(thread_id)` reads `graph.get_state(config).values` and returns the chat messages (Human and AI only, as `{"role", "content"}` dicts) and the saved topic. Mark this function with a comment `# <-- loads an old conversation back into the interface`.

## Gradio interface (`gr.Blocks`)
Title: "Curious Minds Club Chatbot", with one short line of description (pick a club, the bot only talks about that club, conversations are saved).

Controls:
| control | type | behavior |
|---|---|---|
| Club | `gr.Dropdown(TOPICS, value=DEFAULT_TOPIC)` | Changing it keeps the **same** thread. Nothing is sent to the model; the next message uses the new topic. |
| Saved conversations | `gr.Dropdown` of `(label, thread_id)` | Choosing one calls `load_conversation`, shows its messages in the chat window, sets the Club dropdown to its saved topic, and makes it the open thread, so new messages continue it. |
| Refresh list | `gr.Button` | Reloads the Saved conversations choices from the database. |
| New conversation | `gr.Button` | Creates a new `thread_id`, clears the chat window and clears the Saved conversations selection. The Club dropdown keeps its value. |
| Chat window | `gr.Chatbot()` | Shows the open conversation using Gradio 6's default messages format. |
| Message box + Send | `gr.Textbox` and `gr.Button`; Enter also sends | Sends only the new message and the current topic: `graph.invoke({"messages": [HumanMessage(text)], "topic": topic}, config)` with the open `thread_id`, creating one if it is empty. Ignore empty messages. Build the chat window from the invoke result's `result["messages"]`, clear the message box, and refresh the Saved conversations list. |

- Keep the open `thread_id` in `gr.State(new_thread)`, so each browser tab starts with its own fresh conversation. On page load (`demo.load`), fill only the Saved conversations list from the database; if `send_message` receives an empty `thread_id`, create one with `new_thread()` before calling the graph.
- Add 3 example questions under the message box (for example "What are the moons of Mars?", "How do I make pancakes?", "Who won the 2022 World Cup?").
- Launch with `demo.launch(debug=True, show_error=True)`. Add a markdown note: the printed `gradio.live` link is public and uses my OpenAI key, so it must not be shared, and the cell should be stopped when done.

## Code style
- Clear markdown headers between sections: setup, secrets, model, topics and system message, state, node, graph, checkpointer, drawing, graph test, conversations, interface.
- Short docstrings on functions.
- No key, password or database URI anywhere in the code or outputs.

---

## Revision log
- v1: initial spec, based on the class prompts (checkpointer + Gradio): Supabase `PostgresSaver`, `topic` in the state, system message built from the topic on every call, threads listed from the database, and a `gr.Blocks` interface.
- v2: first run in Colab failed with `OpenAIError: Missing credentials` in the model cell, because `OPENAI_API_KEY` was only set later, in the checkpointer cell. Moved loading the key into its own Secrets cell right after the install cell, before `init_chat_model`.
- v3: the Gradio cell never finished and printed nothing. `list_conversations()` called `graph.get_state` inside the `for checkpoint in checkpointer.list(None)` loop; `PostgresSaver.list` holds the connection lock while it yields, so the inner call waited forever. Split it into two steps: first collect all thread ids, then read each thread's state.
- v4: the deadlock fix worked (the conversation list loaded), but building the interface failed with `TypeError` on `gr.Chatbot(type="messages")`: the installed Gradio 6 no longer accepts `type`. Removed the argument; the chat still uses `{"role", "content"}` dicts.
- v5: the interface opened, but every Send showed "Error" on the chat window, the message box and the saved list. Reproduced locally: with `gr.State()` the open `thread_id` can be `None`, the turn is saved under a different thread, and `get_state(config).values["messages"]` raises `KeyError`. Fixed with `gr.State(new_thread)`, a fallback that creates a `thread_id` if it is empty, building the chat from the `invoke` result, and `demo.launch(debug=True, show_error=True)` so errors are visible.
- v6: trying to break the bot showed it was too strict, not too loose: in the Cooking club it refused "As a football coach, what should I eat before a game?" (a question about food) because it mentions football. Off-topic and old-topic questions, and essay requests disguised as Python or geography tasks, were all refused correctly. Added a rule to `SYSTEM_TEMPLATE`: judge a question by what it actually asks, not by the words it contains.
