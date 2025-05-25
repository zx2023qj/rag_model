from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
import uvicorn

# 配置
app = FastAPI()
encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
POS_WINDOW = 3

# Neo4j 驱动
driver = GraphDatabase.driver("bolt://ip:port", auth=("neo4j", "password"))

# 请求/响应模型
class SearchRequest(BaseModel):
    query: str
    book_id: str = None
    top_k: int = 5
    use_rerank: bool = False  # 新增控制是否重排序的参数
    deep_search: bool = True # 是否增加检索
    expand_window: int = 1


class SearchResult(BaseModel):
    chunk_id: str = None
    start_pos: int = None
    end_pos: int = None
    page_number: int = None
    text: str
    similarity: float = None
    rerank_score:float = None



@app.post("/search", response_model=list[SearchResult])
def search(request: SearchRequest):
    try:
        # 编码查询
        query_vector = encoder.encode(request.query).tolist()
        # print(query_vector)
        
        # 使用 Neo4j 原生向量索引检索
        with driver.session() as session:
            result = session.run(
                """
                CALL db.index.vector.queryNodes(
                  'chunk_vector_index',
                  $top_k_estimate,
                  $query_vector
                  
                )
                YIELD node, score
                WHERE node.book_id IS NULL OR node.book_id = $book_id
                RETURN node.chunk_id as chunk_id,node.start_pos AS start_pos,node.end_pos AS end_pos,node.page_number AS page,node.text AS text, score AS similarity
                ORDER BY similarity DESC
                """,
                query_vector=query_vector,
                book_id=request.book_id,
                top_k_estimate=request.top_k * 3
            )
            # print(result)
            chunks = [{
                "chunk_id": record["chunk_id"],
                "start_pos": record["start_pos"],
                "end_pos": record["end_pos"],
                "page_number": record["page"],
                "text": record["text"],
                "similarity": record["similarity"]
            } for record in result]
            print(chunks)
        
        # 重排序（可选）
        if request.use_rerank and chunks:
            pairs = [(request.query, chunk["text"]) for chunk in chunks]
            rerank_scores = reranker.predict(pairs)
            for chunk, score in zip(chunks, rerank_scores):
                chunk["rerank_score"] = float(score)
            chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

    

        if request.deep_search and chunks:
            expand_window = request.expand_window
            retrieved_chunks = chunks[:request.top_k]
            expanded_texts = []
            # min_page = 0xfffffff
            # max_page = -1
            for chunk in retrieved_chunks:
                # 获取当前页的邻近段落
                book_id = chunk["chunk_id"][:5] + '_'
                page = chunk["page_number"]
                start_chunk_pos = max(1, chunk["start_pos"] - expand_window)
                end_chunk_pos = chunk["end_pos"] + expand_window
                # if page >= max_page: 
                #     max_page = page
                # if page <= min_page:
                #     min_page = page
                # 从数据库查询邻近 Chunks
                with driver.session() as session:
                    result = session.run("""
                        MATCH (c:Chunk)
                        WHERE c.chunk_id STARTS WITH $book_id 
                        AND c.page_number = $page
                        AND c.start_pos >= $start
                        AND c.end_pos <= $end
                        RETURN c.text
                        ORDER BY c.start_pos
                    """, {
                        "book_id": book_id,
                        "page": page,
                        "start": start_chunk_pos,
                        "end": end_chunk_pos
                    })
                    expanded_texts.extend([record["c.text"] for record in result])
                    expanded_text = "\n\r".join(set(expanded_texts))
                    print(expanded_text)
            # return [{"start_pos": min_page,"end_pos": max_page,"text": expanded_text}]
            return [{"text": expanded_text}]
        
        return chunks[:request.top_k]
            
    except Exception as e:
        raise HTTPException(500, detail=str(e))

# 健康检查
@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="10.33.68.193", port=8000)