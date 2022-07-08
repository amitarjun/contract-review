from haystack.nodes import FARMReader, TransformersReader
# In-Memory Document Store
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import TextConverter, PDFToTextConverter, DocxToTextConverter, PreProcessor
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import TfidfRetriever
import os
from haystack.utils import clean_wiki_text, convert_files_to_docs, print_answers

class predictions:

    def __init__(self, model_path:str):
        self.reader = FARMReader(model_name_or_path=model_path, use_gpu=True)

    def get_prediction(questions:list, contract:str):
        """The function takes three arguments: questions (list of questions), model (path to the model), and contract (path to the contract folder)"""
        document_store = InMemoryDocumentStore()
        all_docs = convert_files_to_docs(dir_path="contracts", clean_func=clean_wiki_text, split_paragraphs=True)
        document_store.write_documents(all_docs)
        retriever = TfidfRetriever(document_store=document_store)
        all_predictions = []
        pipe = ExtractiveQAPipeline(self.reader, retriever)
        for each in question_set:
            prediction = pipe.run(
                query=each, params={"Retriever": {"top_k": 1}, "Reader": {"top_k": 1}}
            )
            all_predictions.append(prediction)

        return all_predictions


