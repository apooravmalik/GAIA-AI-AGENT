import os 
from dotenv import load_dotenv 
from supabase.client import Client, create_client
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import SupabaseVectorStore 
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool 
load_dotenv()

supabase: Client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"]
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_store = SupabaseVectorStore(
    client=supabase,
    embedding= embeddings,
    table_name="docs",
    query_name="match_documents"
)

all_rows = supabase.table("docs").select("content").execute().data

qa_dict: dict[str, str] = {}
for row in all_rows:
    raw = row["content"]
    if "Answer:" in raw:
        parts = raw.split("Answer:", 1)
        question_part = parts[0].strip()
        answer_part = parts[1].strip()
        if question_part.lower().startswith("question"):
            question_part = question_part.split(":", 1)[1].strip()
        qa_dict[question_part] = answer_part
    else:
        qa_dict[raw.strip()] = ""


@tool
def find_answer(query: str) -> str:
    """
    1) Do an exact dict lookup if possible (not shown here).
    2) Otherwise, run an embedding search and extract ONLY the final‐answer line.
       We look for “Final answer :” or “Answer:” and return just that tail.
    """
    # … (you may have an in‐memory dict check here) …

    # If you fall back to embedding search:
    similar_docs = vector_store.similarity_search(query, k=1)
    if not similar_docs:
        return "Sorry, I couldn’t find that question in my database."

    full_content = similar_docs[0].page_content
    # Look for “Final answer :” first
    if "Final answer :" in full_content:
        # Extract everything after the first occurrence of “Final answer :”
        answer_text = full_content.split("Final answer :", 1)[1].strip()
        return answer_text

    # Fallback if they used “Answer:” instead
    if "Answer:" in full_content:
        answer_text = full_content.split("Answer:", 1)[1].strip()
        return answer_text

    # If neither tag exists, just return whatever is after the last newline
    # (or the entire text, but without the question prefix).
    lines = full_content.strip().splitlines()
    return lines[-1].strip()
# def find_answer(query: str) -> str:
#     """
#     If 'query' exactly matches a key in qa_dict, return qa_dict[query].
#     Otherwise, do an embedding search (k=1) in Supabase and return only the "Answer:" portion. 
#     """

#     if query in qa_dict:
#         return qa_dict[query]
#     similar_docs = vector_store.similarity_search(query, k=1)
#     if not similar_docs:
#         return "Sorry, I couldn't find that question"
#     top_doc = similar_docs[0].page_content
#     if "Answer:" in top_doc:
#         return top_doc.split("Answer:", 1)[1].strip()
#     if "Final answer: " in top_doc:
#         return top_doc.split("Final answer :", 1)[1].strip()
#     return top_doc.strip()

tools = [find_answer]

def build_graph():
    """
    Build a LangGraph where every HumanMessage is handled by find_answe(---),
    and the returned AIMessage contains exactly the stored answer text.    
    """
    def retriever_node(state: MessagesState):
        user_query = state["messages"][-1].content
        answer_text = find_answer(user_query)
        return {"messages": state["messages"] + [AIMessage(content=answer_text)]}
    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever_node)
    builder.set_entry_point("retriever")
    builder.set_finish_point("retriever")
    return builder.compile()