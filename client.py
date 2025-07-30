import argparse
import requests
import os
import sys

def auto_detect_genre(file_path: str) -> str:
    """Try to detect genre from filename (before first dot)."""
    filename = os.path.basename(file_path)
    if "." in filename:
        return filename.split(".")[0]
    return ""

def send_file(url: str, file_path: str, genre: str = "") -> dict:
    """Send a single file to the API and return prediction."""
    with open(file_path, "rb") as file:
        files = {"file": (os.path.basename(file_path), file, "audio/wav")}
        data = {}
        if genre:
            data["actual_genre"] = genre

        try:
            response = requests.post(url, files=files, data=data)
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    if response.status_code == 200:
        try:
            result = response.json()
            return {
                "predicted": result.get("predicted_genre", "Unknown"),
                "status": "ok"
            }
        except ValueError:
            return {"error": "Invalid JSON response"}
    else:
        return {"error": f"HTTP {response.status_code}: {response.text}"}

def main():
    parser = argparse.ArgumentParser(description="Send audio file(s) to the prediction API.")
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:80/predict",
        help="Prediction API URL (default: http://127.0.0.1:80/predict)"
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the audio file or folder containing files."
    )
    parser.add_argument(
        "--genre",
        type=str,
        default="",
        help="Actual genre label (optional). If not provided, tries to detect from filename."
    )
    args = parser.parse_args()

    # Prepare list of files
    if os.path.isfile(args.file):
        files_to_send = [args.file]
    elif os.path.isdir(args.file):
        files_to_send = [
            os.path.join(args.file, f)
            for f in os.listdir(args.file)
            if f.lower().endswith((".wav", ".mp3", ".flac"))
        ]
        if not files_to_send:
            print(f"No audio files found in folder: {args.file}")
            sys.exit(1)
    else:
        print(f"Error: Path not found: {args.file}")
        sys.exit(1)

    # Process files
    results = []
    for file_path in files_to_send:
        # Determine genre
        genre = args.genre.strip()
        if not genre:
            genre = auto_detect_genre(file_path)
            if genre:
                print(f"[INFO] Auto-detected genre for '{os.path.basename(file_path)}': '{genre}'")
            else:
                print(f"[INFO] No genre detected for '{os.path.basename(file_path)}'.")

        # Send file to API
        print(f"[INFO] Sending file: {file_path}")
        result = send_file(args.url, file_path, genre)
        if "error" in result:
            print(f"  [ERROR] {result['error']}")
        else:
            print(f"  [RESULT] Predicted genre: {result['predicted']}")
        results.append((file_path, genre, result))

    # Summary
    print("\n=== Summary ===")
    for fpath, genre, res in results:
        if "error" in res:
            print(f"{os.path.basename(fpath)} | Actual: {genre or 'N/A'} | ERROR: {res['error']}")
        else:
            print(f"{os.path.basename(fpath)} | Actual: {genre or 'N/A'} | Predicted: {res['predicted']}")

if __name__ == "__main__":
    main()


