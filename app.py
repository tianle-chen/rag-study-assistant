"""Run with: streamlit run app.py"""
import hashlib
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import APIError
from pydantic import ValidationError

from study_assistant.core import DemoEmbeddings, StudyError, StudyIndex, parse_pdf

load_dotenv()
st.set_page_config(page_title='知页 Pagewise · 学习助手', page_icon=':material/menu_book:', layout='wide')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;600&family=Noto+Serif+SC:wght@500;700&display=swap');
html, body, .stApp {font-family:'Noto Sans SC',sans-serif;}
h1, h2, h3 {font-family:'Noto Serif SC',serif !important;}
[data-testid="stSidebar"] {border-right:1px solid #DCD8CC;}
[data-testid="stChatMessage"] {border:1px solid #DCD8CC;border-radius:6px;}
.stButton>button[kind="primary"] {background:#245746;border-color:#245746;}
.eyebrow {font-size:12px;letter-spacing:.24em;color:#A35B29;font-weight:600;}
.note {border-left:3px solid #A35B29;padding:4px 0 4px 16px;color:#57645C;}
</style>''', unsafe_allow_html=True)

for key, default in [('index', None), ('messages', []), ('signature', None), ('document', '')]:
    if key not in st.session_state:
        st.session_state[key] = default


def clear_index():
    if st.session_state.index is not None:
        st.session_state.index.close()
    st.session_state.index = None
    st.session_state.signature = None
    st.session_state.messages = []
    st.session_state.document = ''


def draw_answer(message):
    st.markdown(message['text'])
    st.caption(message['mode'] + f" · 召回 {message['candidates']} 个候选片段")
    for i, doc in enumerate(message['sources'], 1):
        with st.expander(f"[{i}] {doc.metadata['source']} · PDF 第 {doc.metadata['page']} 页"):
            st.text(doc.page_content)
    st.caption('引用编号与原文片段绑定；请展开证据核对答案。')


with st.sidebar:
    st.markdown('### 知页 / Pagewise')
    st.caption('把厚教材，读成自己的知识。')
    st.divider()
    mode = st.radio('工作模式', ['离线演示', 'OpenAI 问答'])
    key = ''
    model = os.getenv('OPENAI_CHAT_MODEL', 'gpt-4.1-mini')
    embedding_model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
    if mode == 'OpenAI 问答':
        key = st.text_input('OpenAI API Key', type='password', value=os.getenv('OPENAI_API_KEY', ''))
        model = st.text_input('回答 / 重排序模型', value=model)
        embedding_model = st.text_input('Embedding 模型', value=embedding_model)
        consent = st.checkbox('允许将教材文本发送至 OpenAI（会产生 API 费用）')
        st.caption('密钥只在当前服务进程内使用，请勿将其上传至 GitHub。')
    else:
        consent = False
        st.caption('本地词汇检索与原文摘录，无 API 费用；不具备模型语义理解能力。')
    uploaded = st.file_uploader('上传教材 PDF', type=['pdf'])
    use_sample = st.checkbox('使用示例教材《数据库复习手册》', value=uploaded is None)
    with st.expander('检索设置'):
        chunk_size = st.slider('分块字符数', 200, 1000, 500, 50)
        overlap = st.slider('重叠字符数', 0, 150, 50, 10)
        fetch_k = st.slider('初次召回候选数', 4, 24, 12)
        top_k = st.slider('重排序后 Top-K', 1, min(fetch_k, 8), min(4, fetch_k))
    build = st.button('建立 / 更新知识库', type='primary', use_container_width=True)
    if st.button('清空教材与对话', use_container_width=True):
        clear_index()
        st.rerun()
    st.caption('仅支持可提取文本的 PDF · ≤30 MB · ≤600 页。页码使用 PDF 物理页序号。')

st.markdown('<div class="eyebrow">READ / RETRIEVE / UNDERSTAND</div>', unsafe_allow_html=True)
st.title('从一页证据，理解一个知识点。')
st.markdown('<div class="note">上传教材，提出问题。每个回答，都能回到原文。</div>', unsafe_allow_html=True)
st.write('')

# A configuration change invalidates old results immediately, preventing cross-mode/document answers.
data = None
filename = ''
if use_sample:
    filename = '数据库复习手册.pdf'
    data = (Path(__file__).parent / 'samples' / 'database-notes.pdf').read_bytes()
elif uploaded is not None:
    filename, data = uploaded.name, uploaded.getvalue()
signature = hashlib.sha256(repr((
    hashlib.sha256(data).hexdigest() if data else '', filename, mode,
    hashlib.sha256(key.encode()).hexdigest(), model, embedding_model,
    chunk_size, overlap, consent,
)).encode()).hexdigest()
if st.session_state.signature and signature != st.session_state.signature:
    clear_index()
    st.info('教材或模型设置已变化，请重新建立知识库。')

if build:
    if not data:
        st.warning('请上传 PDF 或选择示例教材。')
    elif mode == 'OpenAI 问答' and (not key or not consent):
        st.warning('请填写 API Key，并确认教材文本发送范围。')
    else:
        try:
            with st.spinner('正在解析教材、分块并建立向量索引…'):
                pages = parse_pdf(data, filename)
                if mode == 'OpenAI 问答':
                    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
                    embeddings = OpenAIEmbeddings(api_key=key, model=embedding_model,
                                                  request_timeout=45, max_retries=1)
                    llm = ChatOpenAI(api_key=key, model=model, timeout=60, max_retries=1)
                else:
                    embeddings, llm = DemoEmbeddings(), None
                new_index = StudyIndex(pages, embeddings, llm, chunk_size, overlap)
                clear_index()
                st.session_state.index = new_index
                st.session_state.signature = signature
                st.session_state.document = filename
            st.success('知识库已就绪，可以开始提问。')
        except StudyError as exc:
            st.error(str(exc))
        except (APIError, ValidationError, ValueError, RuntimeError, OSError):
            st.error('索引建立失败。请检查 API 密钥、额度、模型名称和网络连接后重试。')

index = st.session_state.index
if index:
    a, b, c = st.columns([3, 1, 1])
    a.metric('当前教材', st.session_state.document)
    b.metric('有效文本页', index.pages)
    c.metric('知识片段', len(index.chunks))
else:
    st.info('从左侧选择示例教材并点击「建立 / 更新知识库」，或上传自己的 PDF。')
    st.markdown('**可以这样问**')
    st.write('事务的 ACID 特性分别是什么？　 / 　主键和外键有什么区别？　 / 　索引有什么优缺点？')

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        if message['role'] == 'user':
            st.markdown(message['text'])
        else:
            draw_answer(message)

question = st.chat_input('提出一个完整问题，例如：事务的 ACID 特性分别是什么？', disabled=index is None, max_chars=2000)
if question and index:
    st.session_state.messages.append({'role': 'user', 'text': question})
    with st.chat_message('user'):
        st.markdown(question)
    with st.chat_message('assistant'):
        try:
            with st.spinner('正在检索、重排序并整理证据…'):
                answer = index.ask(question, k=top_k, fetch_k=fetch_k)
            message = {'role': 'assistant', 'text': answer.text, 'sources': answer.sources,
                       'mode': answer.mode, 'candidates': answer.candidates}
            st.session_state.messages.append(message)
            draw_answer(message)
        except StudyError as exc:
            st.error(str(exc))
        except (APIError, ValidationError, ValueError, RuntimeError, OSError):
            st.error('问答失败。请检查 API 额度、模型是否支持结构化输出及网络连接后重试。')

st.divider()
st.caption('知页 Pagewise · 每次提问独立检索，请使用完整问题。关闭会话后索引不持久保存；共享部署由服务端处理上传内容。')
