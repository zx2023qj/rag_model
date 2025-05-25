import pdfplumber
import spacy
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

# 1. 读取PDF
def extract_pdf_content(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        chunks = []
        # 使用spacy分句
        nlp = spacy.load("zh_core_web_sm")
        for idx,page in tqdm(enumerate(pdf.pages), total=len(pdf.pages), desc="Processing PDF"):
            text = page.extract_text()
            meta = {
                "page_number": page.page_number,
                "page_width": page.width,
                "page_height": page.height
            }
            doc = nlp(text)
            sentences = [sent.text.replace('\n', ' ').replace('\r', ' ').strip() for sent in doc.sents]
            # 合并为段落（每段3句）
            for i in range(0, len(sentences), 3):
                chunk_text = "".join(sentences[i:i+3])
                chunks.append({
                    "text": chunk_text,
                    "page_number": meta["page_number"],
                    "start_pos": i,
                    "end_pos": i+3
                })
        
        print('pdf数据提取完毕')

        # with open("../output.json", "w", encoding="utf-8") as f:
        #     json.dump(chunks, f, ensure_ascii=False, indent=2)
        #     print("数据已保存到 output.json")
        return chunks
    

# 2. 生成向量
def generate_vectors(chunks):
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    vectors = model.encode([chunk["text"] for chunk in chunks], batch_size=32)
    for i, chunk in enumerate(chunks):
        vectors[i] = [x for x in vectors[i]]
        chunk["vector"] = vectors[i].tolist()

    print('向量索引生成完毕')

    # with open("../output_vector.json", "w", encoding="utf-8") as f:
    #     json.dump(chunks, f, ensure_ascii=False, indent=2)
    #     print("数据已保存到 output_vector.json")
    return chunks

# 3. 存储到Neo4j
def save_to_neo4j(chunks, book_meta):
    driver = GraphDatabase.driver("bolt://192.168.35.129:7687", auth=("neo4j", "albert-thermos-button-palma-frozen-4437"))
    with driver.session() as session:
        # 创建Book节点
        session.run("""
            CREATE (b:Book {
                book_id: $book_id,
                title: $title,
                author: $author
            })
        """, book_meta)
        
        # 创建Chunk节点及关系
        for chunk in chunks:
            session.run("""
                MATCH (b:Book {book_id: $book_id})
                CREATE (c:Chunk {
                    chunk_id: $chunk_id,
                    text: $text,
                    page_number: $page_number,
                    start_pos: $start_pos,
                    end_pos: $end_pos,
                    vector: $vector
                })
                CREATE (c)-[:BELONGS_TO]->(b)
            """, {
                "book_id": book_meta["book_id"],
                "chunk_id": f"{book_meta['book_id']}_chunk_{chunk['page_number']}_{chunk['start_pos']}",
                **chunk
            })
        print("neo4j 结构创建完成")

def create_vector_index():
    driver = GraphDatabase.driver("bolt://192.168.35.129:7687", auth=("neo4j", "albert-thermos-button-palma-frozen-4437"))
    with driver.session() as session:
        # 创建vector的索引
        session.run(
            """CREATE VECTOR INDEX chunk_vector_index 
            FOR (c:Chunk) ON (c.vector) OPTIONS
            { indexConfig: { `vector.dimensions`: 384,`vector.similarity_function`: 'cosine'}}"""
        )
    print("vector索引创建完成")

def remake_neo4j():
    driver = GraphDatabase.driver("bolt://192.168.35.129:7687", auth=("neo4j", "albert-thermos-button-palma-frozen-4437"))
    with driver.session() as session:
        # remake!!!
        session.run(
            """drop index chunk_vector_index"""
        )
        session.run(
            """MATCH (n) DETACH DELETE n"""
        )                
    print("删除数据库完成")

# # 主流程
pdf_path = "../demo.pdf"
book_meta = {
    "book_id": "book1",
    "title": "机电系统故障诊断与维修案例教程",
    "author": "段向军"
}

chunks = extract_pdf_content(pdf_path)
chunks_with_vectors = generate_vectors(chunks)
save_to_neo4j(chunks_with_vectors, book_meta)
create_vector_index()

# remake_neo4j()
