import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json

from openai import OpenAI
from dotenv import load_dotenv
import requests
from config import IndexerConfig as Config

load_dotenv()

import logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

sample_json = {
    "key_entities": ["emily e. roberts", "michael thompson"],
}

query_metadata_prompt = """
Analyze the provided query and extract structured insights with a focus on type safety and specificity for legal and financial documents.

<query>
{query}
</query>

Return the results as a JSON object with the following fields, ensuring type safety and using lowercase for all values:

- "key_entities" (list of strings): A list of significant entities mentioned in the query (corporations, individuals, organizations, etc.).

** Guidelines **
1. If any initials are already present in the query, add dot if it is missing. For example, "john d. rockefeller" instead of "john d rockefeller".
2. If nothing is present, give an empty list in the respective JSON field. 
3. Ensure all text values in the JSON output are in lowercase.
4. Fix any spelling mistakes if present.

Example input:
With respect to the Residential Lease Agreement between Emily E Roberts and Michael Thompson, what are the key terms and conditions?

Example output:
{sample_json}

Return only the JSON object in the specified format.
"""

choose_entities_prompt = """
You are an expert at choosing relevant key entities according to the query. \
Given the query and key entities from a source, choose the most relevant key entities based on the query.
It is not necessary for the related entity to be explicitly mentioned in the query.

<query>
{query}
</query>

<entities>
{entities}
</entities>

**Guidelines**
1. Return only the MOST relevant key entities as a list of strings.
2. Do not include any additional information in the output.
3. If there are multiple similar entities, include all of them.

Example input:
query: "What are the key terms and conditions of the Residential Lease Agreement between Emily E Roberts and Michael Thompson?"
topics: ["lease agreement", "residential lease", "terms and conditions", "emily e roberts", "michael thompson"]

Example output:
["emily e roberts", "michael thompson"]

Return only the list of strings in the specified format.
"""

choose_topics_prompt = """
You are an expert at choosing relevant topics according to the query. \
Given the query and the topics extracted from the documents, choose the most relevant topics based on the query.

<query>
{query}
</query>

<topics>
{topics}
</topics>

**Guidelines**
1. Return only the MOST relevant topics as a list of strings.
2. Do not include any additional information in the output.
3. Do not give more than 10 topics.

Example input:
query: "What are the key terms and conditions of the Residential Lease Agreement between Emily E Roberts and Michael Thompson?"
topics: ["lease agreement", "residential lease", "terms and conditions", "emily e roberts", "michael thompson"]

Example output:
["lease agreement", "residential lease", "key terms"]

Return only the list of strings in the specified format.
"""

class MetadataFilterGenerator:
    
    def __init__(self):
        config = Config()
        self.vector_db_url = config.VECTOR_DB_URL
        self.inputs_url = config.VECTOR_DB_URL + "/v1/inputs"
        self.metadatas = []
        self.key_entities = []
        self.topics = []
        self.wlibl_thresh = 0.7
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"

    def _get_max_common_substring_ratio(self, str1, str2) -> float:
        """
        Get the ratio of max common substring with the ratio of the length.
        """

        max_len = 0
        for i in range(len(str1)):
            for j in range(i + 1, len(str1)):
                if str1[i:j] in str2:
                    max_len = max(max_len, j - i)
        
        ratio1 = max_len / len(str1)
        ratio2 = max_len / len(str2)

        return max(ratio1, ratio2)
    
    def _update_metadatas(self):
        """
        Get the list of entities from the metadata.
        """
        try:
            doc_inputs = requests.get(self.inputs_url)
            self.metadatas = doc_inputs.json()
            self.key_entities = set()
            for metadata in self.metadatas:
                for entity in metadata["key_entities"]:
                    self.key_entities.add(entity)
        except Exception as e:
            logger.error(f"Error getting entities: {str(e)}")      


    # def _get_entities(self, query) -> list[str]:
    #     self._update_metadatas()
    #     query_metadata = self._get_query_metadata(query)
    #     logger.info(str(query_metadata))
    #     query_entities = query_metadata["key_entities"]
    #     if not query_entities:
    #         return []
    #     max_score = {"score": 0, "entity": None}
    #     entities = []
    #     for key_entity in self.key_entities:
    #         if len(key_entity) < 4:
    #             continue
    #         for query_entity in query_entities:
    #             if len(query_entity) < 4:
    #                 continue
    #             score = self._get_max_common_substring_ratio(key_entity, query_entity)
    #             if score > self.wlibl_thresh:
    #                 entities.append(key_entity)
    #             if score > max_score["score"]:
    #                 max_score["score"] = score
    #                 max_score["entity"] = key_entity

    #     for key_entity in self.key_entities:
    #         if len(key_entity) >= 4:
    #             continue
    #         for query_entity in query_entities:
    #             if len(query_entity) >= 4:
    #                 continue
    #             if key_entity == query_entity:
    #                 entities.append(key_entity)
        
    #     logger.info(f"Max score: {str(max_score)}")

    #     return entities
            
    def _get_entities(self, query: str) -> list[str]:
        """
        Get the relevant entities based on the query using an LLM Call.
        """
        self._update_metadatas()
        
        content = choose_entities_prompt.format(query=query, entities=str(self.key_entities))
        response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=1000,
                    temperature=0.0,
                )
        return eval(response.choices[0].message.content)


    
    def _get_entity_docs(self, entities: list[str]) -> list[str]:
        """
        Get the documents wit any overlapping entities.
        """
        doc_ids = set()
        for document in self.metadatas:
            for entity in entities:
                if entity in document["key_entities"]:
                    doc_ids.add(document["id"])
        return list(doc_ids)

    def _get_topics(self, doc_ids: list[str]) -> list[str]:
        """
        Get the topics for the documents.
        """
        topics = set()
        for document in self.metadatas:
            for relevant_id in doc_ids:
                if document["id"] == relevant_id:
                    for sub_doc_id in document["topics"]:
                        for topic in document["topics"][sub_doc_id]:
                            topics.add(topic)

        return list(topics)
    
    def _get_relevant_topics(self, query, topics: list[str]) -> list[str]:
        """
        Get the relevant topics based on the query using an LLM Call.
        """
        
        content = choose_topics_prompt.format(query=query, topics=str(topics))
        response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=1000,
                    temperature=0.0,
                )
        return eval(response.choices[0].message.content)

    def _get_query_metadata(self, query: str) -> dict:
        """
        Get the metadata for the query based on the metadata.
        """
        key_entities = []

        content = query_metadata_prompt.format(query=query, sample_json=json.dumps(sample_json, indent=4))
        response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=1000,
                    temperature=0.0,
                )
        metadata = response.choices[0].message.content
        metadata = metadata.replace("```json", "").replace("```", "")
        metadata = json.loads(metadata)
        key_entities = metadata["key_entities"]
        return {
            "key_entities": key_entities,
        }

    def get_filter(self, query: str, useTopic:bool = True) -> list[str]:
        """
        Get the filter for the query based on the metadata.
        """
        logging.info(f"Getting metadata filter for: {query}")
        relevant_entities = self._get_entities(query)
        logger.info(f"Relevant entities found: {str(relevant_entities)}")
        doc_ids = self._get_entity_docs(relevant_entities)
        logger.info(f"Doc ids found: {str(doc_ids)}")
        entity_filter = " || ".join([f"id == `{doc_id}`" for doc_id in doc_ids])

        if not useTopic:
            return entity_filter

        
        topics = self._get_topics(doc_ids)
        logger.info(str(topics))
        # return entity_filter, topics
        relevant_topics = self._get_relevant_topics(query, topics)
        print("Relevant topics: ", relevant_topics)
        topic_filter = " || ".join([f"contains(topics, `{topic}`)" for topic in relevant_topics])

        return f"({entity_filter}) && ({topic_filter})"
        

if __name__ == "__main__":

    metadata_generator = MetadataRetriever()
    query = "Are there any product categories / service categories that represent more than 20% of Boeing's revenue for FY2022?"
    filter = metadata_generator.get_filter(query)
    print("Extracted filter: ", filter)

