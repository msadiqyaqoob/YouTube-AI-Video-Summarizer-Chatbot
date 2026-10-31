from pathlib import Path
import sys
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "core"))

# Imports
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title

from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions
)

# Load .env
load_dotenv(BASE_DIR / ".env", override=True)


source = "https://www.youtube.com/watch?v=XN3xNJvWXsc&t=245s"

# english = Whisper
# hinglish = Sarvam
language = "english"


# STEP 1: Audio Processing
print("\nSTEP 1: Processing audio...")

chunks = process_input(source)

print("Chunks created:", len(chunks))


# STEP 2: Transcription
print("\nSTEP 2: Starting transcription...")

transcription = transcribe_all(
    chunks,
    language=language
)

print("\nTRANSCRIPTION COMPLETED!")


# Save transcript
with open(BASE_DIR / "transcription.txt", "w", encoding="utf-8") as file:
    file.write(transcription)


# STEP 3: Summarization
print("\nSTEP 3: Generating summary...")

summary = summarize(transcription)

print("\nSUMMARY:")
print(summary)


# STEP 4: Generate Title
print("\nSTEP 4: Generating title...")

title = generate_title(transcription)

print("\nTITLE:")
print(title)


# STEP 5: Extract Action Items
print("\nSTEP 5: Extracting action items...")

action_items = extract_action_items(transcription)

print("\nACTION ITEMS:")
print(action_items)


# STEP 6: Extract Key Decisions
print("\nSTEP 6: Extracting key decisions...")

decisions = extract_key_decisions(transcription)

print("\nKEY DECISIONS:")
print(decisions)


# STEP 7: Extract Open Questions
print("\nSTEP 7: Extracting open questions...")

questions = extract_questions(transcription)

print("\nOPEN QUESTIONS:")
print(questions)


# STEP 8: Save All Results
print("\nSTEP 8: Saving results...")

with open(BASE_DIR / "summary.txt", "w", encoding="utf-8") as file:

    file.write("TITLE:\n")
    file.write(title)

    file.write("\n\nSUMMARY:\n")
    file.write(summary)

    file.write("\n\nACTION ITEMS:\n")
    file.write(action_items)

    file.write("\n\nKEY DECISIONS:\n")
    file.write(decisions)

    file.write("\n\nOPEN QUESTIONS:\n")
    file.write(questions)


print("\nALL TASKS COMPLETED SUCCESSFULLY!")
