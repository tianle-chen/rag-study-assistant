import io
from pathlib import Path

import pytest
from langchain_core.documents import Document
from pypdf import PdfWriter

from study_assistant.core import (
    MAX_BYTES,
    DemoEmbeddings,
    Fact,
    GroundedAnswer,
    RerankResult,
    StudyError,
    StudyIndex,
    lexical_rerank,
    parse_pdf,
    split_pages,
)


@pytest.fixture
def index():
    idx = StudyIndex([
        Document(page_content='ACID includes atomicity, consistency, isolation and durability.',
                 metadata={'source': 'textbook.pdf', 'page': 2}),
        Document(page_content='Indexes accelerate queries but cost storage and write performance.',
                 metadata={'source': 'textbook.pdf', 'page': 3}),
    ], DemoEmbeddings())
    yield idx
    idx.close()


def test_real_pdf_pages_and_chinese():
    data = (Path(__file__).parents[1] / 'samples/database-notes.pdf').read_bytes()
    pages = parse_pdf(data, '../notes.pdf')
    assert len(pages) == 4
    assert '原子性' in pages[1].page_content
    assert pages[1].metadata == {'source': 'notes.pdf', 'page': 2}


def test_invalid_pdf():
    with pytest.raises(StudyError, match='有效'):
        parse_pdf(b'not a pdf', 'x.pdf')


def test_size_limit():
    with pytest.raises(StudyError, match='30 MB'):
        parse_pdf(b'%PDF-' + b'0' * MAX_BYTES, 'x.pdf')


def test_scanned_pdf_message():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    stream = io.BytesIO()
    writer.write(stream)
    with pytest.raises(StudyError, match='OCR'):
        parse_pdf(stream.getvalue(), 'scan.pdf')


def test_encrypted_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt('password')
    stream = io.BytesIO()
    writer.write(stream)
    with pytest.raises(StudyError, match='密码'):
        parse_pdf(stream.getvalue(), 'locked.pdf')


def test_page_limit():
    writer = PdfWriter()
    for _ in range(601):
        writer.add_blank_page(width=100, height=100)
    stream = io.BytesIO()
    writer.write(stream)
    with pytest.raises(StudyError, match='600'):
        parse_pdf(stream.getvalue(), 'long.pdf')


def test_chunking_never_loses_page():
    chunks = split_pages([Document(page_content='数据库' * 500,
                                  metadata={'source': 'a.pdf', 'page': 7})])
    assert len(chunks) > 1
    assert all(len(d.page_content) <= 500 and d.metadata['page'] == 7 for d in chunks)
    assert len({d.metadata['chunk_id'] for d in chunks}) == len(chunks)


def test_invalid_overlap():
    with pytest.raises(StudyError):
        split_pages([], chunk_size=100, overlap=100)


def test_bm25_is_separate_ranking():
    docs = [Document(page_content='ACID atomicity'), Document(page_content='index index query')]
    assert lexical_rerank('index', docs, 1)[0] == docs[1]
    assert lexical_rerank('nonexistent', docs, 1) == []


def test_vector_retrieval_and_page_citation(index):
    result = index.ask('ACID atomicity', k=1, fetch_k=2)
    assert '[1]' in result.text
    assert result.sources[0].metadata['page'] == 2
    assert result.candidates == 2


def test_unknown_question_refuses(index):
    assert index.ask('xyzunknown').sources == []


@pytest.mark.parametrize('question', ['', ' ' * 5, 'a' * 2001])
def test_invalid_question(index, question):
    with pytest.raises(StudyError):
        index.ask(question)


def test_session_isolation(index):
    other = StudyIndex([Document(page_content='Secret private bananas',
                                metadata={'source': 'private.pdf', 'page': 1})], DemoEmbeddings())
    try:
        assert all(d.metadata['source'] == 'textbook.pdf'
                   for d in index.store.similarity_search('bananas', k=5))
        assert other.store._collection.count() == 1
        assert index.store._collection.count() == 2
    finally:
        other.close()


class FakeLLM:
    def __init__(self, facts, ids=None):
        self.facts = facts
        self.ids = [1, 1, 999] if ids is None else ids
        self.calls = []

    def with_structured_output(self, schema):
        outer = self

        class Runner:
            def invoke(self, messages):
                outer.calls.append(messages)
                if schema is RerankResult:
                    return RerankResult(ids=outer.ids)
                return GroundedAnswer(facts=outer.facts)
        return Runner()


def test_live_pipeline_contract_with_mock_model(index):
    llm = FakeLLM([Fact(text='事务具有原子性。', citations=[1])])
    index.llm = llm
    answer = index.ask('ACID atomicity')
    assert answer.text == '事务具有原子性。 [1]'
    assert len(answer.sources) == 1
    assert len(llm.calls) == 2
    assert 'untrusted' in llm.calls[0][0].content


@pytest.mark.parametrize('citations', [[], [999], [0], [-1]])
def test_unknown_citation_rejected(index, citations):
    index.llm = FakeLLM([Fact(text='unsupported', citations=citations)])
    answer = index.ask('ACID')
    assert 'unsupported' not in answer.text
    assert '未通过' in answer.text


def test_model_can_decline_retrieval(index):
    index.llm = FakeLLM([], ids=[])
    assert index.ask('ACID').sources == []
