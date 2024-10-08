# app/knowledge_base/integrated_kb_query.py

import logging
from typing import List, Dict, Any
import json
import re
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.indices.keyword_table import KeywordTableIndex
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import Settings, PropertyGraphIndex, PromptTemplate
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.selectors import (
    PydanticMultiSelector,
    PydanticSingleSelector,
)
from llama_index.core.response_synthesizers import TreeSummarize
from dotenv import load_dotenv
import numpy as np
import nest_asyncio
import os
from flask import session

load_dotenv()
nest_asyncio.apply()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IntegratedKnowledgeBaseQuery:
    def __init__(self):
        self.embed_model, self.llm = self._initialize_components()
        self.graph_store = self._setup_graph_store()
        self.vector_store = self._setup_vector_store()
        self.graph_index, self.vector_index = self._setup_index()

    def _initialize_components(self):
        embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.embed_model = embed_model
        llm = Groq(model="llama3-70b-8192", api_key=os.getenv('GROQ_API_KEY'), temperature=0)
        Settings.llm = llm

        return embed_model, llm

    def _setup_graph_store(self):
        return Neo4jPropertyGraphStore(
            url = os.getenv('NEO4J_URL'),
            username = "neo4j",
            password = os.getenv('NEO4J_PASSWORD'),
            database="neo4j",
            refresh_schema=False,
            sanitize_query_output=True
        )

    def get_neo4j_schema(self):
        cypher_query = """
        CALL db.schema.visualization()
        """
        try:
            result = self.graph_store.structured_query(cypher_query)
            return result
        except Exception as e:
            logging.error(f"Error retrieving Neo4j schema: {str(e)}")
            return None

    def _setup_vector_store(self):
        return QdrantVectorStore(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name="legislative_docs",
        )

    def _setup_index(self):
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            graph_store=self.graph_store,
        )
        graph_index = PropertyGraphIndex.from_existing(
            property_graph_store=self.graph_store,
            storage_context=storage_context)

        vector_index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            storage_context=storage_context,
        )
        return graph_index, vector_index

    def diagnose_stores(self):
        logging.info("Diagnosing graph store...")
        self._diagnose_graph_store()
        logging.info("Diagnosing vector store...")
        self._diagnose_vector_store()

    def _diagnose_graph_store(self):
        query = "MATCH (n) RETURN count(n) as node_count"
        result = self.graph_store.structured_query(query)
        node_count = result[0]['node_count']
        logging.info(f"Total nodes in the graph: {node_count}")

        query = "MATCH (n) RETURN DISTINCT labels(n) as node_types"
        result = self.graph_store.structured_query(query)
        node_types = [r['node_types'][0] for r in result if r['node_types']]
        logging.info(f"Node types in the graph: {', '.join(node_types)}")

        query = "MATCH (n) RETURN n LIMIT 5"
        result = self.graph_store.structured_query(query)
        logging.info("Sample nodes:")
        for record in result:
            logging.info(record['n'])

    def _diagnose_vector_store(self):
        collection_info = self.vector_store.client.get_collection(collection_name="legislative_docs")
        logging.info(f"Vector store collection info: {collection_info}")

    def query_graph_store(self, query: str) -> List[Dict[str, Any]]:
        entities = re.findall(r'\b[A-Z][a-z]+ (?:[A-Z][a-z]+ )*(?:Co\.|Corporation|Inc\.|LLC)\b|\b[A-Z][a-z]+\b', query)
        logging.info(f"Entities found: {entities}")
        cypher_query = """
        MATCH (e)
        WHERE e.name CONTAINS $entity_name
        OPTIONAL MATCH (e)-[r]-(related)
        RETURN e as entity, type(r) as relationship_type, related
        LIMIT 5
        """

        graph_results = []
        for entity in entities:
            params = {"entity_name": entity}
            try:
                results = self.graph_store.structured_query(cypher_query, params)
                graph_results.extend(results)
            except Exception as e:
                logging.error(f"Error querying graph store for entity '{entity}': {str(e)}")

        return graph_results

    def get_bill_details(self, bill_id: str) -> List[Dict[str, Any]]:
        cypher_query = """
        MATCH (b:Bill {id: $bill_id})
        OPTIONAL MATCH (b)-[:AMENDS]->(a:Act)
        OPTIONAL MATCH (b)-[:CONTAINS]->(p:Provision)
        OPTIONAL MATCH (b)-[:DEFINES]->(d:Definition)
        OPTIONAL MATCH (b)-[:INVOLVED]->(person:Person)
        OPTIONAL MATCH (b)-[:RELATES_TO]->(act:Act)
        OPTIONAL MATCH (b)-[:AFFECTS]->(affected)
        RETURN b as bill,
              collect(DISTINCT a) as amendments,
              collect(DISTINCT p) as provisions,
              collect(DISTINCT d) as definitions,
              collect(DISTINCT person) as persons_involved,
              collect(DISTINCT act) as related_acts,
              collect(DISTINCT {type: labels(affected)[0], details: properties(affected)}) as affected_entities
        """
        results = self.graph_store.structured_query(cypher_query, {"bill_id": bill_id})
        return results


    def format_graph_results(self, query: str) -> tuple:
        graph_results = self.query_graph_store(query)
        logging.info(f"format_graph_results function: {graph_results}")
        if not graph_results:
            return "No relevant information found in the graph database.", []

        formatted_results = []
        bill_details = []
        for result in graph_results:
            entity = result['entity']
            relationship = result['relationship_type']
            related = result['related']

            formatted_result = f"{entity.get('type', 'Entity')} {entity.get('name', 'Unknown')}"
            if relationship and related:
                related_name = related.get('name', 'Unknown')
                formatted_result += f"\n  {relationship}: {related_name}"

            formatted_results.append(formatted_result)

            if entity.get('type') == 'Bill':
                bill_detail = self.get_bill_details(entity.get('name'))
                if bill_detail:
                    formatted_bill = self.format_bill_details(bill_detail)
                    bill_details.append(formatted_bill)

        return "\n\n".join(formatted_results), bill_details

    def format_vector_results(self, query) -> str:
        query_vector = self.embed_model.get_text_embedding(query)
        vector_results = self.vector_store.client.search(
            collection_name="legislative_docs",
            query_vector=query_vector,
            limit=2
        )
        formatted_results = []
        for i, result in enumerate(vector_results, 1):
            payload = result.payload
            score = result.score
            node_content = json.loads(result.payload['_node_content'])
            content = node_content.get('text', '')
            formatted_results.append(f"Document {i} (Score: {score:.4f}):\n{content[:300]}...")
        return "\n".join(formatted_results)

    def generate_llm_response(self, query: str, response: str, graph_results: str, vector_results: str, bill_details: List[str]) -> str:
        prompt = f"""You are a highly knowledgeable Legal AI assistant specializing in analyzing legislative documents and bills. Your task is to provide a short and accurate response to the following query based on the information from both a graph database and a vector database.

        Query: {query}

        Initial Response: {response}

        Knowledge Context: {graph_results} + {vector_results}

        Bill Details: {bill_details}

        Instructions:
        1. Analyze all provided contexts, extracting all relevant information related to the query.
        2. Provide a clear, concise, and well-structured response that directly addresses the query.
        3. Include specific details such as bill numbers, dates, amendments, key provisions, and related entities when available in any context.
        4. If the contexts contain information about multiple related bills or legal issues, summarize each one briefly and explain their relevance to the query.
        5. If there are any conflicting opinions or interpretations in the contexts, present them objectively and explain the implications.
        6. Use legal terminology accurately, but also provide explanations for complex terms to ensure clarity.
        7. If the contexts don't provide sufficient information to fully answer the query, clearly state what is known and what information is missing.
        8. Do not make assumptions or include information not present in the given contexts.
        9. Conclude your response with a brief summary of the key points.
        10. After your main response, suggest two follow-up questions that would be relevant for further exploration of the topic, prefaced with "For further exploration, you might consider asking:".

        Remember to maintain an objective, professional tone throughout your response. Do not refer to the query or contexts directly in your answer; instead, incorporate the information seamlessly into your response.

        Now, based on these instructions, please provide your comprehensive analysis and response."""

        llm_output = self.llm.complete(prompt).text
        return llm_output
        
    def query_knowledge_base(self, query: str) -> str:
        logging.info(f"Querying knowledge base: {query}")
        
        # Access previous context from the session
        previous_context = session.get('history', [])

        # Determine strategy based on query and context
        if self.is_new_query(query, previous_context):
            response = self.query_datastores(query)
        elif self.can_use_previous_context(query, previous_context):
            response = self.use_previous_context(query, previous_context)
        else:
            response = self.combine_context_and_query(query, previous_context)
        
        # Store response in session for future context
        if 'history' not in session:
            session['history'] = []
        session['history'].append(response)

        return response

    def is_new_query(self, query: str, context: list) -> bool:
        return not any(entity in query for entity in context)

    def can_use_previous_context(self, query: str, context: list) -> bool:
        return any(entity in query for entity in context)

    def combine_context_and_query(self, query: str, context: list) -> str:
        datastore_response = self.query_datastores(query)
        combined_response = self.llm.complete(f"{datastore_response} with additional insights from previous interactions.").text
        return combined_response

    def use_previous_context(self, query: str, context: list) -> str:
        relevant_context = [c for c in context if c in query]
        llm_output = self.llm.complete(relevant_context).text
        return llm_output

    def query_datastores(self, query: str) -> str:
        # Create query engines
        graph_query_engine = self.graph_index.as_query_engine()
        vector_query_engine = self.vector_index.as_query_engine()

        # Create tools
        graph_tool = QueryEngineTool.from_defaults(
            query_engine=graph_query_engine,
            description="Useful for answering questions about relationships and connections between entities",
        )

        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            description="Useful for answering detailed questions about legal content",
        )

        TREE_SUMMARIZE_PROMPT_TMPL = (
            """You are a helpful legal AI assistant specialized in understanding the legislative enquiries"""
        )
        
        tree_summarize = TreeSummarize(
            summary_template=PromptTemplate(TREE_SUMMARIZE_PROMPT_TMPL)
        )

        # Create router query engine
        router_query_engine = RouterQueryEngine(
            selector=LLMMultiSelector.from_defaults(),
            query_engine_tools=[graph_tool, vector_tool],
            summarizer=tree_summarize,
        )

        # Execute query
        response = router_query_engine.query(query)

        graph_results, case_details = self.format_graph_results(query)
        logging.info(f"Formatted graph results: {graph_results}")

        vector_results = self.format_vector_results(query)
        logging.info(f"Formatted vector results: {vector_results}")

        llm_response = self.generate_llm_response(query, str(response), graph_results, vector_results, case_details)
        
        logging.info(f"LLM response: {llm_response}")

        return str(llm_response)
