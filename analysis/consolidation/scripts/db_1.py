import chromadb


class Database:
    db: chromadb.Client
    collection: chromadb.Collection

    def __init__(self, db_path):
        self.db = chromadb.PersistentClient()
        self.collection = self.db.get_or_create_collection(db_path)

    def insert_document(self, document):
        self.collection.insert(document)

    def insert_document_by_embedding(self, id, embeddings, document):
        self.collection.add(ids=id, embeddings=embeddings, documents=document)

    def get_document(self, document_id):
        return self.collection.get(document_id, include=["documents", "embeddings"])

    def search_nearest_documents(self, embeddings, n):
        return self.collection.query(
            embeddings,
            n_results=n,
            include=["documents", "embeddings", "distances"],
        )

    def delete_document(self, document_id):
        self.collection.delete(document_id)

    def update_document(self, document_id, new_document):
        self.collection.update(document_id, new_document)

    def list_collections(self):
        return self.db.list_collections()

    def clear_collection(self):
        self.collection.delete(where_document={"$contains": {"text": " "}})

    def close(self):
        self.db.close()
