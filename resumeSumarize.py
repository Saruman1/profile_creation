from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from pdfminer.high_level import extract_text
import torch
import re

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf(resume_sections, filename):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    for section, entities in resume_sections.items():
        story.append(Paragraph(f"<b>{section}:</b>", styles['Heading2']))
        if section == 'Summary':
            story.append(Paragraph(entities[0], styles['BodyText']))  # Summary
        else:
            story.append(Paragraph(", ".join(set(entities)), styles['BodyText']))  # Other sections
        story.append(Spacer(1, 12))  # Add some space between sections

    doc.build(story)

# Load summarization and NER models
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small", device=0)
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", tokenizer="dslim/bert-base-NER", aggregation_strategy="simple", device=0)

# Extract text from your resume
text = extract_text(r'E:\python\pythonProjectAlgorythm\Lib\Resume.pdf')

# Summarize the entire resume text
summary = summarizer(text, max_length=200, min_length=50, do_sample=False)[0]['summary_text']

# Perform Named Entity Recognition (NER) on the extracted text
ner_results = ner_pipeline(text)

# Mapping of entity types to resume sections
entity_to_section = {
    'PER': 'Summary',
    'ORG': 'Work Experience & Relevant Projects',
    'LOC': 'Summary',
    'MISC': 'Skills & Certifications',
}

# Group entities by section
resume_sections = {'Summary': [summary]}
for ent in ner_results:
    entity_type = ent['entity_group']
    word = ent['word']
    resume_sections.setdefault(entity_to_section.get(entity_type, 'Other'), []).append(word)  # Group entities

# Print the results in the desired format
for section, entities in resume_sections.items():
    print(f"\n{section}:")
    if section == 'Summary':
        print(entities[0])  # Print the summary directly
    else:
        print(", ".join(set(entities)))  # Print unique entities in other sections

# Generate the PDF after processing the resume
generate_pdf(resume_sections, 'resume_summary.pdf')
