import openai
import json

embedding_model = openai.OpenAI()


def get_text_embedding(text: str, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return (
        embedding_model.embeddings.create(input=[text], model=model).data[0].embedding
    )


def get_entity_description_embedding(name: str, description: str):
    combined_text = f"{name}: {description}"
    return get_text_embedding(combined_text)


def get_entity_metadata_embedding(metadata: dict):
    combined_text = json.dumps(metadata)
    return get_text_embedding(combined_text)
