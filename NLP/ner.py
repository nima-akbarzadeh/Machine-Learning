import spacy
import re


def extract_entities_spacy(text):
    """
    Extract named entities from a text using SpaCy.

    Parameters:
    text (str): The input text.

    Returns:
    list of tuple: A list of tuples where each tuple contains an entity and its label.
    """

    doc = spacy_model_en(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]

    return entities


def extract_entities_scratch(text):
    """
    Extract named entities from a text using a simple rule-based approach.

    Parameters:
    text (str): The input text.

    Returns:
    list of tuple: A list of tuples where each tuple contains an entity and its label.
    """
    entities = []

    # Find all names
    for match in name_pattern.finditer(text):
        entities.append((match.group(), "PERSON"))

    # Find all locations
    for match in location_pattern.finditer(text):
        entities.append((match.group(), "GPE"))

    # Find all organizations
    for match in organization_pattern.finditer(text):
        entities.append((match.group(), "ORG"))

    return entities


if __name__ == '__main__':

    # load language model
    spacy_model_en = spacy.load('en_core_web_sm')
    # Sample data for named entity recognition
    names = ["Apple", "Google"]
    locations = ["U.K.", "United Kingdom", "USA", "New York"]
    organizations = ["Apple", "Google", "Microsoft", "Amazon"]

    # Compile regular expressions for the entity categories
    name_pattern = re.compile(r'\b(' + '|'.join(names) + r')\b')
    location_pattern = re.compile(r'\b(' + '|'.join(locations) + r')\b')
    organization_pattern = re.compile(r'\b(' + '|'.join(organizations) + r')\b')

    # Example usage
    text = "Apple is looking at buying U.K. startup for $1 billion."
    entities = extract_entities_spacy(text)
    print(entities)
    entities = extract_entities_scratch(text)
    print(entities)
