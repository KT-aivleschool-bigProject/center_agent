import os
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import langgraph
from langgraph.graph import StateGraph, END
import json
import re

# 환경변수 로드
env_loaded = load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")  # 예: 'octocat/Hello-World'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def extract_json_from_codeblock(text: str) -> str:
    # ```json ... ``` 또는 ``` ... ``` 코드블록 제거
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        return match.group(1)
    return text


# --- LLM 기반 프롬프트 분류 ---
def classify_prompt(user_prompt: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 분류기입니다. 사용자의 요청이 코드에서 특정 기능을 찾는 요청이면 'find_code', PR 요약 요청이면 'pr_summary', 코드 리뷰/설명이면 'code_review', 그 외는 'other'로만 답하세요. 반드시 json으로 {{\"type\": \"...\"}} 형식으로 답하세요.",
            ),
            ("human", "{user_prompt}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0)
    chain = prompt | llm
    result = chain.invoke({"user_prompt": user_prompt})
    import json

    try:
        json_str = extract_json_from_codeblock(result.content)
        type_str = json.loads(json_str)["type"]
        print(type_str)
    except Exception as e:
        print(f"분류 LLM 응답 오류: {e}, content: {result.content}")
        type_str = "other"
    return type_str


# --- LLM 기반 기능 키워드 추출 ---
def extract_feature_from_prompt(prompt: str) -> str:
    if not OPENAI_API_KEY:
        return prompt
    feature_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                '아래 사용자의 요청에서 핵심 기능 키워드(예: 로그인, 회원가입, 인증 등)만 한 단어로 추출해서 json으로 {{"feature": "..."}} 형식으로 답하세요. 만약 적절한 기능 키워드가 없으면 전체 프롬프트를 그대로 반환하세요.',
            ),
            ("human", "{user_prompt}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0)
    chain = feature_prompt | llm
    result = chain.invoke({"user_prompt": prompt})
    import json

    try:
        json_str = extract_json_from_codeblock(result.content)
        feature = json.loads(json_str)["feature"]
    except Exception as e:
        print(f"기능 추출 LLM 응답 오류: {e}, content: {result.content}")
        feature = prompt
    return feature


# --- GitHub 코드 검색 ---
def github_code_search(query, repo_full_name, github_token):
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }
    url = f"https://api.github.com/search/code"
    params = {"q": f"{query} repo:{repo_full_name}", "per_page": 3}
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    items = resp.json().get("items", [])
    results = []
    for item in items:
        file_path = item["path"]
        html_url = item["html_url"]
        raw_url = html_url.replace("github.com", "raw.githubusercontent.com").replace(
            "/blob/", "/"
        )
        code_snippet = ""
        try:
            code_resp = requests.get(raw_url)
            if code_resp.ok:
                code_snippet = code_resp.text[:1000]
        except Exception:
            pass
        results.append((file_path, html_url, code_snippet))
    return results


def find_code_for_feature_github(feature, repo_full_name, github_token):
    results = github_code_search(feature, repo_full_name, github_token)
    if not results:
        return f"'{feature}' 기능을 포함하는 코드를 찾지 못했습니다."
    response = f"'{feature}' 기능이 포함된 파일:\n"
    for file_path, html_url, code_snippet in results:
        response += f"\n--- [{file_path}]({html_url}) ---\n"
        response += f"```java\n{code_snippet}\n```\n"
    response += "\n(코드는 일부만 표시됩니다)"
    return response


# --- PR 요약/설명 ---
def fetch_latest_pr() -> dict:
    if not GITHUB_TOKEN or not GITHUB_REPO:
        raise ValueError(
            "GITHUB_TOKEN 또는 GITHUB_REPO 환경변수가 설정되어 있지 않습니다."
        )
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    url = f"https://api.github.com/repos/{GITHUB_REPO}/pulls?state=all&sort=created&direction=desc&per_page=1"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    pr_list = resp.json()
    if not pr_list:
        raise ValueError("열린 PR이 없습니다.")
    pr = pr_list[0]
    diff_url = pr["url"] + "/files"
    diff_resp = requests.get(diff_url, headers=headers)
    diff_resp.raise_for_status()
    files = diff_resp.json()
    diff_summary = ""
    for f in files:
        filename = f.get("filename")
        patch = f.get("patch", "")
        diff_summary += f"--- {filename} ---\n{patch}\n\n"
    return {
        "title": pr["title"],
        "body": pr.get("body", ""),
        "diff": diff_summary,
        "author": pr["user"]["login"],
        "number": pr["number"],
    }


def summarize_pr(pr: dict) -> str:
    summary = f"PR 제목: {pr['title']}\n"
    summary += f"작성자: {pr['author']}\n"
    summary += f"본문: {pr['body']}\n"
    summary += f"변경된 코드(diff):\n{pr['diff']}\n"
    return summary


def explain_pr(pr: dict) -> str:
    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY 환경변수가 필요합니다."
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 소프트웨어 엔지니어입니다. 아래 PR의 변경사항을 한국어로 친절하게 설명해 주세요.",
            ),
            ("human", "PR 제목: {title}\n본문: {body}\n변경된 코드(diff):\n{diff}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.2)
    chain = prompt | llm
    result = chain.invoke(
        {"title": pr["title"], "body": pr["body"], "diff": pr["diff"][:4000]}
    )
    return result.content if hasattr(result, "content") else str(result)


# --- 코드 리뷰 ---
def explain_code_with_llm(code: str) -> str:
    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY 환경변수가 필요합니다."
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 소프트웨어 엔지니어입니다. 아래 코드를 한국어로 친절하게 설명하거나 리뷰해 주세요.",
            ),
            ("human", "{code}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.2)
    chain = prompt | llm
    result = chain.invoke({"code": code[:4000]})
    return result.content if hasattr(result, "content") else str(result)


# --- LangGraph 챗봇 그래프 노드 정의 ---
def entry_node(state: dict) -> dict:
    state["prompt_type"] = classify_prompt(state["user_input"])
    return state


def find_code_node(state: dict) -> dict:
    feature = extract_feature_from_prompt(state["user_input"])
    state["result"] = find_code_for_feature_github(feature, GITHUB_REPO, GITHUB_TOKEN)
    return state


def pr_summary_node(state: dict) -> dict:
    pr = fetch_latest_pr()
    summary = summarize_pr(pr)
    explanation = explain_pr(pr)
    state["result"] = f"### PR 요약\n{summary}\n\n### PR 설명\n{explanation}"
    return state


def code_review_node(state: dict) -> dict:
    # state["code"]에 코드가 있다고 가정
    state["result"] = explain_code_with_llm(state.get("code", ""))
    return state


def output_node(state: dict) -> dict:
    # Streamlit에서는 return만, CLI에서는 print도 가능
    return state


# --- LangGraph 챗봇 그래프 정의 ---
graph = StateGraph(dict)
graph.add_node("entry", entry_node)
graph.add_node("find_code", find_code_node)
graph.add_node("pr_summary", pr_summary_node)
graph.add_node("code_review", code_review_node)
graph.add_node("output", output_node)


def route_from_entry(state: dict):
    if state["prompt_type"] == "find_code":
        return "find_code"
    elif state["prompt_type"] == "pr_summary":
        return "pr_summary"
    elif state["prompt_type"] == "code_review":
        return "code_review"
    else:
        state["result"] = (
            "지원하지 않는 요청입니다. '기능 코드 찾기', 'PR 요약', '코드 리뷰' 요청만 가능합니다."
        )
        return "output"


graph.add_conditional_edges("entry", route_from_entry)
graph.add_edge("find_code", "output")
graph.add_edge("pr_summary", "output")
graph.add_edge("code_review", "output")
graph.add_edge("output", END)
graph.set_entry_point("entry")


def run_chatbot_graph(user_input: str, code: str = "") -> str:
    state = {"user_input": user_input, "code": code}
    app = graph.compile(checkpointer=None)
    result_state = app.invoke(state)
    return result_state.get("result", "오류: 결과가 없습니다.")


# CodeAgent 클래스 - main.py에서 사용하기 위한 래퍼
class CodeAgent:
    def __init__(self):
        self.name = "Code Agent"

    async def process(self, message: str) -> str:
        """코드 관련 요청 처리 - LangGraph 기반"""
        return run_chatbot_graph(message)
