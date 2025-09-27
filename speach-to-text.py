import os
import argparse
import assemblyai as aai
from dotenv import load_dotenv

def main():
    load_dotenv()
    api_key = os.getenv("AAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing AAI_API_KEY in .env")

    aai.settings.api_key = api_key
    transcriber = aai.Transcriber()

    parser = argparse.ArgumentParser(description="Transcribe audio via AssemblyAI")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="Path to a local audio/video file (wav, mp3, mp4, etc.)")
    src.add_argument("--url", help="Publicly accessible audio URL")
    parser.add_argument("--output", default="transcription.txt", help="Where to save text")
    args = parser.parse_args()

    source = args.file or args.url
    print(f"Transcribing from source: {source}")
    transcript = transcriber.transcribe(source)

    if transcript.status == "error":
        raise SystemExit(f"Transcription failed: {transcript.error}")

    print("\n=== Transcription Result ===\n")
    print(transcript.text)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(transcript.text or "")
    print(f"\nTranscription saved to {args.output}")

if __name__ == "__main__":
    main()
