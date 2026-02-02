import spacy
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


nlp = spacy.load("en_core_web_sm")


text = (
    "Atharva Chinchole, a software engineer working at Google India, "
    "recently attended an AI conference in New York. "
    "During the event, Google announced new advancements in Artificial Intelligence. "
    "Atharva shared updates on Twitter and LinkedIn, mentioning his visit to Microsoft "
    "headquarters in Seattle and meetings with engineers from Amazon and Meta."
)

print("Input Text:")
print(text)
print("=" * 80)


doc = nlp(text)

predicted_entities = [(ent.text, ent.label_) for ent in doc.ents]

print("Predicted Named Entities:")
for ent in predicted_entities:
    print(ent)

print("=" * 80)


true_entities = [
    ("Atharva Chinchole", "PERSON"),
    ("Google India", "ORG"),
    ("New York", "GPE"),
    ("Google", "ORG"),
    ("Twitter", "ORG"),
    ("LinkedIn", "ORG"),
    ("Microsoft", "ORG"),
    ("Seattle", "GPE"),
    ("Amazon", "ORG"),
    ("Meta", "ORG")
]

print("True Named Entities:")
for ent in true_entities:
    print(ent)

print("=" * 80)


true_labels = []
pred_labels = []

for entity, label in true_entities:
    true_labels.append(label)
    if (entity, label) in predicted_entities:
        pred_labels.append(label)
    else:
        pred_labels.append("O")   


accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, average="macro", zero_division=0)
recall = recall_score(true_labels, pred_labels, average="macro", zero_division=0)
f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)

print("Evaluation Metrics:")
print(f"Accuracy  : {accuracy:.2f}")
print(f"Precision : {precision:.2f}")
print(f"Recall    : {recall:.2f}")
print(f"F1-Score  : {f1:.2f}")
