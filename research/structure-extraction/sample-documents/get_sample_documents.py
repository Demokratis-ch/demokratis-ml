r"""Download sample documents (preferably DRAFTs) for the latest consultation for each political body.

If we don't know which documents are DRAFTs, we download all documents for the given consultation.

Manual review is required to delete irrelevant non-DRAFT documents. Also, some of the selected
consultation may not have any DRAFT documents!

Usage:

    uv run \
        research/structure-extraction/sample-documents/get_sample_documents.py \
        data/dataframes/consultation-documents-preprocessed-2024-11-26.parquet
"""  # noqa: INP001

import hashlib
import pathlib
import sys

import httpx
import pandas as pd

SAMPLES_DIRECTORY = pathlib.Path(__file__).parent
UA = "Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"


def _download_documents(df_docs: pd.DataFrame, destination: pathlib.Path) -> tuple[int, int, int]:
    downloads = caches = failures = 0
    for _, document in df_docs.iterrows():
        local_path = destination / _generate_local_path(document, "pdf")
        assert document["document_source_url"].startswith(("http:", "https:"))
        print("downloading", document["document_source_url"])
        print("to         ", local_path)
        if local_path.exists():
            print(f"✅ {local_path} already exists")
            caches += 1
        else:
            try:
                response = httpx.get(
                    document["document_source_url"],
                    headers={"User-Agent": UA},
                )
                response.raise_for_status()
                local_path.write_bytes(response.content)
            except Exception as e:  # noqa: BLE001
                print("❌", repr(e))
                failures += 1
            else:
                print("✅ downloaded")
                downloads += 1
    return (downloads, caches, failures)


def _generate_local_path(
    document: pd.Series,
    extension: str,
    suffix: str = "",
) -> pathlib.Path:
    url_hash = hashlib.sha1(document["document_source_url"].encode()).hexdigest()  # noqa: S324
    return pathlib.Path(
        f"{document['document_id']}-{document['document_language']}-{document['document_type']}"
        f"-{url_hash}{suffix}.{extension}"
    )


if __name__ == "__main__":
    data_file = pathlib.Path(sys.argv[1])
    df = pd.read_parquet(data_file)

    latest_consultations = (
        df.sort_values("consultation_start_date").groupby("political_body").agg({"consultation_id": "last"})
    )
    latest_docs = df[df["consultation_id"].isin(latest_consultations["consultation_id"])]

    downloads = caches = failures = 0
    for consultation_id, docs in latest_docs.groupby("consultation_id"):
        political_body = docs["political_body"].iloc[0]
        consultation_start_date = docs["consultation_start_date"].iloc[0]
        directory = (
            SAMPLES_DIRECTORY / political_body / f"consultation_{consultation_id}_{consultation_start_date:%Y-%m-%d}"
        )
        if directory.exists():
            print(
                f"\n\n=== {political_body}: consultation {consultation_id} already downloaded"
                "and presumably reviewed => skipping"
            )
            continue

        directory.mkdir(parents=True, exist_ok=True)
        drafts = docs[docs["document_type"] == "DRAFT"]
        if drafts.empty:
            print(
                f"\n\n=== {political_body}: no drafts identified for consultation {consultation_id}, "
                "writing all documents"
            )
            drafts = docs
        else:
            print(f"\n\n=== {political_body}: {len(drafts)} drafts")

        counts = _download_documents(drafts, directory)
        downloads += counts[0]
        caches += counts[1]
        failures += counts[2]

    del latest_docs["document_content_plain"]
    latest_docs.sort_values(["political_body", "consultation_start_date", "document_id"]).to_csv(
        SAMPLES_DIRECTORY / "sample_document_metadata.csv", index=False
    )

    print(f"\n{downloads} documents downloaded, {caches} already existed, {failures} failed")
