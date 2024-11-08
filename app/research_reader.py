import os
import base64
from typing import List

import cv2
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document
from langchain.schema.messages import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from unstructured.partition.pdf import partition_pdf

load_dotenv(dotenv_path="/workspace/.env.local")

TOP_K = 5

def main():
    st.title("Research Reader")

    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        if 'uploaded_file' not in st.session_state:
            st.session_state.uploaded_file = None

        if st.session_state.uploaded_file is None:
            uploaded_file = st.file_uploader("PDFファイルをアップロードしてください", type=["pdf"])
            if uploaded_file is not None:
                st.session_state.uploaded_file = uploaded_file
                process_and_store_pdf(uploaded_file)
        else:
            st.write(f"現在のファイル: {st.session_state.uploaded_file.name}")
            if st.button("現在のPDFを削除"):
                st.session_state.uploaded_file = None
                st.session_state.vectorstore = None
                st.session_state.messages = []

    if st.session_state.vectorstore is not None:
        # Chat interface
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("質問を入力してください"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                result, metadata_list = generate_output(
                    st.session_state.vectorstore, prompt
                )
                citations = {}
                for i, metadata in enumerate(metadata_list):
                    if (i in [0, TOP_K]) and (metadata["type"] == "Image"):
                        show_fig(metadata["image_base64"])
                        st.markdown(
                            f"**画像引用ページ**: p.{metadata['page_number']}"
                        )
                    if (
                        f'p.{metadata["page_number"]}'
                        not in citations
                    ):
                        citations[
                            f'p.{metadata["page_number"]}'
                        ] = 1
                    else:
                        citations[
                            f'p.{metadata["page_number"]}'
                        ] += 1

                sorted_citations = sorted(
                    citations.items(), key=lambda item: item[1], reverse=True
                )
                st.markdown("---")
                st.markdown("**参考ページ**")
                for citation, _ in sorted_citations:
                    st.markdown(f"- {citation}")
            st.session_state.messages.append({"role": "assistant", "content": result})
    else:
        st.write("PDFファイルをアップロードしてください。")

def process_and_store_pdf(uploaded_file):
    images, texts = process_pdf(uploaded_file)
    st.write("PDFの処理が完了しました。")
    images = add_summary(images)
    st.write("画像の要約が完了しました。")
    vectorstore = build_vectorstore(texts, images)
    st.session_state.vectorstore = vectorstore
    st.write("ベクトルストアの構築が完了しました。")

def build_vectorstore(texts, images):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
    )
    vectorstore = Chroma(
        embedding_function=embeddings, persist_directory=None  # Not persisting to disk
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    add_vectorstore(vectorstore, texts, images, text_splitter)
    return vectorstore

def process_pdf(file):
    # Get Elements
    raw_pdf_elements = partition_pdf(
        file=file,
        languages=["jpn", "eng"],
        strategy="hi_res",
        extract_images_in_pdf=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=True,
        infer_table_structure=True,
    )

    images = []
    texts = []

    el_dict_list = [el.to_dict() for el in raw_pdf_elements]

    for i, el in enumerate(el_dict_list):
        not_table_title = True
        if i < len(el_dict_list) - 1:
            if el["type"] == "Title" and el_dict_list[i + 1]["type"] == "Table":
                el_dict_list[i + 1]["text"] = (
                    el["text"] + "\n" + el_dict_list[i + 1]["text"]
                )
                not_table_title = False
        if el["type"] in ["Image", "Table"]:
            images.append(el)
        elif not_table_title:
            texts.append(el)

    images = del_small_images(images)

    texts = merge_texts(texts)

    return images, texts

def del_small_images(images, max_kb=30):
    over_max_kb_image_list = []
    for image in images:
        if image["type"] == "Image":
            base64_data = image["metadata"]["image_base64"]
            # Remove padding characters
            padding_characters = base64_data.count("=")
            base64_length = len(base64_data) - padding_characters
            # Calculate original data size
            original_size = (base64_length * 3) // 4
            if original_size > max_kb * 1024:
                over_max_kb_image_list.append(image)
        else:
            over_max_kb_image_list.append(image)
    return over_max_kb_image_list

def merge_texts(texts):
    combined_texts = {}

    for text in texts:
        page = text["metadata"]["page_number"]
        words = text["text"]

        if page in combined_texts:
            combined_texts[page] += "\n" + words
        else:
            combined_texts[page] = words

    result = [
        {
            "type": "Text",
            "text": all_words,
            "metadata": {
                "page_number": page,
            },
        }
        for page, all_words in combined_texts.items()
    ]

    return result

def add_summary(images):
    for image in images:
        image["summary"] = image_summarize(image)
    return images

def image_summarize(image):
    if image["type"] == "Table":
        prompt = (
            "表が画像データとして提供されています。"
            "画像データをもとに、この表に含まれる情報をもれなく正確に、日本語の文章で説明してください。"
            "ただし与えられた表の画像が読み取りにくい場合には、次に提供するHTML形式の表を使用して情報をもれなく正確に日本語で説明してください。"
            "表のHTML形式: {html}"
            "また、HTML形式には誤字が含まれている可能性がありますので、以下のテキスト情報を参考にしてください。"
            "テキスト情報: {text}"
        ).format(html=image["metadata"].get("text_as_html", ""), text=image["text"])
    else:
        prompt = "この画像に含まれる情報をもれなく正確に日本語で説明してください。"

    chat = ChatOpenAI(model="gpt-4o")

    img_base64 = image["metadata"]["image_base64"]

    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )

    return msg.content

def add_vectorstore(vectorstore, texts, images, text_splitter):
    chunks = []
    for text in texts:
        chunk_list = text_splitter.split_text(text["text"])
        for chunk in chunk_list:
            chunk_dict = text.copy()
            chunk_dict["text"] = chunk
            chunks.append(chunk_dict)

    for i, chunk in enumerate(chunks):
        vectorstore.add_documents(
            [
                Document(
                    page_content=chunk["text"],
                    metadata={
                        "type": chunk["type"],
                        "page_number": chunk["metadata"]["page_number"],
                        "image_base64": "",
                    },
                )
            ]
        )

    img_chunks = []
    for image in images:
        img_chunk_list = text_splitter.split_text(image["summary"])
        for img_chunk in img_chunk_list:
            img_chunk_dict = image.copy()
            img_chunk_dict["summary"] = img_chunk
            img_chunks.append(img_chunk_dict)

    for i, image in enumerate(img_chunks):
        vectorstore.add_documents(
            [
                Document(
                    page_content=image["summary"],
                    metadata={
                        "type": "Image",
                        "page_number": image["metadata"]["page_number"],
                        "image_base64": image["metadata"]["image_base64"],
                    },
                )
            ]
        )

def generate_output(vectorstore, user_input):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # Regenerate question
    question = regenerate_question(user_input)

    vector_searched = vector_retriever.get_relevant_documents(question)

    texts_retrieved = []
    metadata_list = []
    for doc in vector_searched:
        texts_retrieved.append(doc.page_content)
        metadata_list.append(doc.metadata)

    result = chat_based_on_texts(texts_retrieved, question)
    return result, metadata_list

def regenerate_question(user_input):
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    prompt = ChatPromptTemplate.from_template(
        """
        次の会話とフォローアップの質問を元に、フォローアップの質問を独立した質問として言い換えてください。
        ----------
        会話の履歴: {chat_history}
        ----------
        フォローアップの質問: {follow_up_question}
        ----------
        独立した質問:
        """
    )
    chain = prompt | llm
    follow_up_question = user_input
    chat_history = st.session_state.messages
    args = {"chat_history": chat_history, "follow_up_question": follow_up_question}
    ans = chain.invoke(args)
    return str(ans.content)

def chat_based_on_texts(texts_retrieved, question):
    texts = "\n\n".join(texts_retrieved)
    prompt_text = f"""
        以下の文書に基づいて質問に対する厳密な回答を生成してください。
        ----------
        会話記録を元にユーザーとの会話のキャッチボールを成立させてください。
        ----------
        会話記録{st.session_state.messages}
        ----------
        文書: {texts}
        ----------
        質問: {question}
        """
    chat = ChatOpenAI(model="gpt-4o")
    return st.write_stream(
        chat.stream([HumanMessage(content=[{"type": "text", "text": prompt_text}])])
    )

def show_fig(image_base64):
    img_binary = base64.b64decode(image_base64)
    jpg = np.frombuffer(img_binary, dtype=np.uint8)
    img = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
    st.image(img)

if __name__ == "__main__":
    main()
