import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Pexels image URL list from SH17 metadata JSON files."
    )
    parser.add_argument(
        "metadata_dir",
        type=Path,
        help="Directory containing SH17 metadata JSON files.",
    )
    args = parser.parse_args()

    metadata_dir = args.metadata_dir.expanduser().resolve()
    if not metadata_dir.is_dir():
        raise NotADirectoryError(f"Metadata directory not found: {metadata_dir}")

    urls = []
    for metadata_path in metadata_dir.iterdir():
        if metadata_path.suffix.lower() != ".json":
            continue

        with metadata_path.open("r", encoding="utf-8") as json_file:
            metadata = json.load(json_file)
        urls.append(metadata["src"]["original"])

    output_path = (
        Path(__file__).resolve().parents[2] / "data" / "list_of_all_urls.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(f"{url}\n" for url in urls), encoding="utf-8")

    print(f"{len(urls)} URLs written to {output_path}")


if __name__ == "__main__":
    main()
